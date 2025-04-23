import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsgluedq.transforms import EvaluateDataQuality
from awsglue import DynamicFrame

def sparkSqlQuery(glueContext, query, mapping, transformation_ctx) -> DynamicFrame:
    for alias, frame in mapping.items():
        frame.toDF().createOrReplaceTempView(alias)
    result = spark.sql(query)
    return DynamicFrame.fromDF(result, glueContext, transformation_ctx)
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Default ruleset used by all target nodes with data quality enabled
DEFAULT_DATA_QUALITY_RULESET = """
    Rules = [
        ColumnCount > 0
    ]
"""

# Script generated for node AWS Glue Data Catalog
AWSGlueDataCatalog_node1745299768471 = glueContext.create_dynamic_frame.from_catalog(database="stedi_db", table_name="accelerometer_trusted", transformation_ctx="AWSGlueDataCatalog_node1745299768471")

# Script generated for node AWS Glue Data Catalog
AWSGlueDataCatalog_node1745299771329 = glueContext.create_dynamic_frame.from_catalog(database="stedi_db", table_name="customer_trusted", transformation_ctx="AWSGlueDataCatalog_node1745299771329")

# Script generated for node Join
Join_node1745299854842 = Join.apply(frame1=AWSGlueDataCatalog_node1745299768471, frame2=AWSGlueDataCatalog_node1745299771329, keys1=["user"], keys2=["email"], transformation_ctx="Join_node1745299854842")

# Script generated for node SQL Query
SqlQuery778 = '''
select distinct customername,email,phone,birthday,
serialnumber,registrationdate,lastupdatedate,sharewithresearchasofdate,
sharewithpublicasofdate,sharewithfriendsasofdate
from myDataSource;
'''
SQLQuery_node1745300607534 = sparkSqlQuery(glueContext, query = SqlQuery778, mapping = {"myDataSource":Join_node1745299854842}, transformation_ctx = "SQLQuery_node1745300607534")

# Script generated for node Amazon S3
EvaluateDataQuality().process_rows(frame=SQLQuery_node1745300607534, ruleset=DEFAULT_DATA_QUALITY_RULESET, publishing_options={"dataQualityEvaluationContext": "EvaluateDataQuality_node1745299732227", "enableDataQualityResultsPublishing": True}, additional_options={"dataQualityResultsPublishing.strategy": "BEST_EFFORT", "observations.scope": "ALL"})
AmazonS3_node1745300076508 = glueContext.getSink(path="s3://stedi-human-sr/customer/curated/", connection_type="s3", updateBehavior="UPDATE_IN_DATABASE", partitionKeys=[], enableUpdateCatalog=True, transformation_ctx="AmazonS3_node1745300076508")
AmazonS3_node1745300076508.setCatalogInfo(catalogDatabase="stedi_db",catalogTableName="customer_curated")
AmazonS3_node1745300076508.setFormat("json")
AmazonS3_node1745300076508.writeFrame(SQLQuery_node1745300607534)
job.commit()