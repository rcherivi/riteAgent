from suds.client import Client
import base64
import os
import time
from xml.dom.minidom import parseString
import xml.etree.ElementTree as ET
import csv
from xml.sax._exceptions import SAXException


def bodyXmlParse(dynamicSqlQuery):
    xml_string = f"""<?xml version='1.0' encoding='utf-8'?>
       <dataModel xmlns="http://xmlns.oracle.com/oxp/xmlp" version="2.0" xmlns:xdm="http://xmlns.oracle.com/oxp/xmlp" xmlns:xsd="http://wwww.w3.org/2001/XMLSchema" defaultDataSourceRef="demo">
          <description>
             <![CDATA[undefined]]>
          </description>
          <dataProperties>
             <property name="include_parameters" value="false"/>
             <property name="include_null_Element" value="false"/>
             <property name="include_rowsettag" value="false"/>
             <property name="xml_tag_case" value="upper"/>
             <property name="generate_output_format" value="csv"/>
             <property name="optimize_query_executions" value="true"/>
             <property name="multithread_query_executions" value="true"/>        
             <property name="sql_monitor_report_generated" value="undefined"/>
          </dataProperties>
          <dataSets>
             <dataSet name="RESULT_DATA" type="simple">
                <sql dataSourceRef="ApplicationDB_HCM" nsQuery="true" sp="true" xmlRowTagName="" bindMultiValueAsCommaSepStr="false">
                   <![CDATA[DECLARE
              type refcursor is REF CURSOR;
              xdo_cursor  refcursor;
              var clob;
       BEGIN
         var := :qryStmt;
         OPEN :xdo_cursor FOR SELECT * FROM
    (  {dynamicSqlQuery} ) ;
       END;]]>
            </sql>
         </dataSet>
      </dataSets>
      <output rootName="DATA_DS" uniqueRowName="false">
         <nodeList name="RESULT_DATA"/>
      </output>
      <eventTriggers/>
      <lexicals/>
      <valueSets/>
      <parameters>
         <parameter name="qryStmt" defaultValue="1" dataType="xsd:string" rowPlacement="1">
            <input label="Query1"/>
         </parameter>
         <parameter name="xdo_cursor" dataType="xsd:string" rowPlacement="1">
            <input label="xdo_cursor"/>
             </parameter>
      </parameters>
      <bursting/>
      <display>
         <layouts>
            <layout name="RESULT_DATA" left="284px" top="407px"/>
            <layout name="DATA_DS" left="6px" top="351px"/>
         </layouts>
         <groupLinks/>
      </display>
       </dataModel>"""
    dom = parseString(xml_string)
    formatted_xml = dom.toprettyxml(indent="  ")
    xml_bytes = base64.b64encode(formatted_xml.encode("utf-8")).decode("utf-8")
    return xml_bytes


def get_login_token(lgn, user, pwd):
    token = lgn.service.login(userID=user, password=pwd)
    print(token)
    return token


def create_object_and_report(
    catalog, token, reportData, folder_path, file_name, report
):
    dm = catalog.service.createObjectInSession(
        objectName=file_name,
        objectType="xdm",
        objectData=reportData,
        bipSessionToken=token,
    )
    print("Data_model created : ", dm)
    do = report.service.createReportInSession(
        reportName=file_name + "_rpt",
        bipSessionToken=token,
        updateFlag="false",
        dataModelURL=dm,
    )
    print("Report Created: ", do)
    return dm, do


def schedule_report(schedule, xdo_file, token):
    schedule_request = {
        "bookBindingOutputOption": False,
        "mergeOutputOption": False,
        "notifyHttpWhenFailed": False,
        "notifyHttpWhenSkipped": False,
        "notifyHttpWhenSuccess": False,
        "notifyHttpWhenWarning": False,
        "notifyWhenFailed": False,
        "notifyWhenSkipped": False,
        "notifyWhenSuccess": False,
        "notifyWhenWarning": False,
        "repeatCount": 1,
        "repeatInterval": 0,
        "reportRequest": {
            "attributeFormat": "xml",
            "attributeLocale": "en_US",
            "byPassCache": True,
            "flattenXML": False,
            "reportAbsolutePath": xdo_file,
            "sizeOfDataChunkDownload": -1,
        },
        "saveDataOption": True,
        "saveOutputOption": True,
    }
    schedule_id = schedule.service.scheduleReportInSession(
        scheduleRequest=schedule_request, bipSessionToken=token
    )
    print("Scheduled report with Id", schedule_id)
    time.sleep(15)

    request_history = {
        "filter": {"jobId": schedule_id},
        "beginIdx": 1,
        "bipSessionToken": token,
    }
    schedule_job = schedule.service.getAllScheduledReportHistoryInSession(
        **request_history
    )
    if schedule_job is None:
        time.sleep(15)
        schedule_job = schedule.service.getAllScheduledReportHistoryInSession(
            **request_history
        )

    scheduled_job_id = schedule_job["jobInfoList"]["item"][0]["jobId"]
    print("childId", scheduled_job_id)
    return scheduled_job_id


def job_completion_status(scheduled_job_id, token, schedule):
    ss = schedule.service.getScheduledJobInfoInSession(
        jobInstanceID=scheduled_job_id, bipSessionToken=token
    )

    return ss.status == "Success"


def get_report_definition(token, schedule, scheduled_job_id):
    job_output = schedule.service.getScheduledReportOutputInfoInSession(
        jobInstanceID=scheduled_job_id, bipSessionToken=token
    )
    job_output_Id = job_output["jobOutputList"]["item"][0]["outputId"]
    print("jobOutputId", job_output_Id)

    document_name = schedule.service.downloadDocumentDataInSession(
        jobOutputID=job_output_Id, bipSessionToken=token
    )
    print("document_name", document_name)
    return document_name


def get_output(output_file, token, document_name, report):
    xml_data = b""
    remove_first_line = True
    begin_idx = 1
    while begin_idx != -1:
        result = report.service.downloadReportDataChunkInSession(
            fileID=document_name,
            beginIdx=begin_idx,
            size=100000000,
            bipSessionToken=token,
        )
        decoded_data = base64.b64decode(result.reportDataChunk)
        if remove_first_line:
            xml_data = decoded_data[len(decoded_data.split(b"\n", 1)[0]) + 1 :]
            remove_first_line = False
        else:
            xml_data = decoded_data

        with open(output_file, "ab") as file:
            file.write(xml_data)

        begin_idx = result.reportDataOffset

        root = ET.fromstring(xml_data)

        rows = root.findall(".//ROW")
        columns = [col.tag for col in rows[0]] if rows else []

        csv_file = output_file.replace(".txt", ".csv")
        with open(csv_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([col.text for col in row])

        print("Output saved to", csv_file)


def delete_object_and_report(token, catalog, xdm_file, xdo_file):
    print("Deleting Datamodel and Report")
    catalog.service.deleteObjectInSession(xdo_file, token)
    catalog.service.deleteObjectInSession(xdm_file, token)


def delete_objects_in_session(token):
    # Login and get the token
    lgn = Client(
        r"https://fa-etao-dev20-saasfademo1.ds-fa.oraclepdemos.com/xmlpserver/services/v2/SecurityService?wsdl"
    )

    # Initialize the CatalogService client
    catalog = Client(
        r"https://fa-etao-dev20-saasfademo1.ds-fa.oraclepdemos.com/xmlpserver/services/v2/CatalogService?wsdl"
    )

    # Delete objects in session
    catalog.service.deleteObjectInSession("/AI_Output_0.xdm", token)
    catalog.service.deleteObjectInSession("/AI_Output_0_rpt.xdo", token)


def authenticate(userID, password):
    login = Client(
        r"https://fa-etao-dev20-saasfademo1.ds-fa.oraclepdemos.com/xmlpserver/services/v2/SecurityService?wsdl"
    )

    try:
        token = get_login_token(login, userID, password)
        print(token)
        return token

    except Exception as e:
        print("not working")
        return e


def main(table_name, userID, password, base_folder_path, query_input, batch_size=10000):
    report = Client(
        r"https://fa-etao-dev20-saasfademo1.ds-fa.oraclepdemos.com/xmlpserver/services/v2/ReportService?wsdl"
    )
    schedule = Client(
        r"https://fa-etao-dev20-saasfademo1.ds-fa.oraclepdemos.com/xmlpserver/services/v2/ScheduleService?wsdl"
    )
    catalog = Client(
        r"https://fa-etao-dev20-saasfademo1.ds-fa.oraclepdemos.com/xmlpserver/services/v2/CatalogService?wsdl"
    )
    login = Client(
        r"https://fa-etao-dev20-saasfademo1.ds-fa.oraclepdemos.com/xmlpserver/services/v2/SecurityService?wsdl"
    )
    print(userID)
    print(password)
    token = get_login_token(login, userID, password)
    print(token)
    offset = 0
    total_time_elapsed = 0

    folder_path = os.path.join(base_folder_path, table_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    while True:
        output_file = os.path.join(folder_path, f"{table_name}.txt")
        start_time = time.time()
        query = query_input

        print(query)

        if offset > 9999:
            print("Completed Writing")
            break

        report_data = bodyXmlParse(query)
        xdm_file, xdo_file = create_object_and_report(
            catalog, token, report_data, folder_path, f"{table_name}", report
        )
        schedule_id = schedule_report(schedule, xdo_file, token)

        if not job_completion_status:
            delete_object_and_report(token, catalog, xdm_file, xdo_file)
            print("SQL query failed. Please regenerate SQL query")
        else:
            try:
                # This means that the report is on the way
                report_name = get_report_definition(token, schedule, schedule_id)
                get_output(output_file, token, report_name, report)
                print("Output saved to", output_file)

                delete_object_and_report(token, catalog, xdm_file, xdo_file)
                offset += batch_size
                end_time = time.time()
                print("Time elapsed:", end_time - start_time)
                total_time_elapsed += end_time - start_time

                return True
            except Exception as e:
                print(e.__str__)
                print("There was an error with running the query")
                delete_object_and_report(token, catalog, xdm_file, xdo_file)
                os.rmdir(table_name)
                return False

    print("Total time elapsed:", total_time_elapsed)


if __name__ == "__main__":
    table_name = "AI_Output"
    userID = "lisa.jones"
    password = "o6%2E%Wb"
    base_folder_path = ""
    query = """
    select
        ptxa.document_name,
        ptxa.batch_name,
        pte.message_name,
        ptxa.quantity,
        ptxa.denom_raw_cost,
        ptxa.orig_transaction_reference,
        ppab.segment1 as project_number
        from PJC_TXN_XFACE_ALL ptxa, PJC_TXN_ERRORS pte, pjf_projects_all_b ppab
        where 1=1
        and ptxa.txn_interface_id = pte.source_txn_id
        and ptxa.project_id= ppab.project_id
    """
    main(table_name, userID, password, base_folder_path, query)

    # token = authenticate(userID=userID, password=password)

    # if isinstance(token, Exception):
    #     print(True)
    # else:
    #     print(False)
