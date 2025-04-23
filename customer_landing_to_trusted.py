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
AWSGlueDataCatalog_node1745296850314 = glueContext.create_dynamic_frame.from_catalog(database="stedi_db", table_name="customer_landing", transformation_ctx="AWSGlueDataCatalog_node1745296850314")

# Script generated for node SQL Query
SqlQuery545 = '''
select * from myDataSource 
where sharewithreasearchasofdate is not null

'''
SQLQuery_node1745296856903 = sparkSqlQuery(glueContext, query = SqlQuery545, mapping = {"myDataSource":AWSGlueDataCatalog_node1745296850314}, transformation_ctx = "SQLQuery_node1745296856903")

# Script generated for node Amazon S3
EvaluateDataQuality().process_rows(frame=SQLQuery_node1745296856903, ruleset=DEFAULT_DATA_QUALITY_RULESET, publishing_options={"dataQualityEvaluationContext": "EvaluateDataQuality_node1745296837790", "enableDataQualityResultsPublishing": True}, additional_options={"dataQualityResultsPublishing.strategy": "BEST_EFFORT", "observations.scope": "ALL"})
AmazonS3_node1745296866666 = glueContext.getSink(path="s3://stedi-human-sr/customer/trusted/", connection_type="s3", updateBehavior="UPDATE_IN_DATABASE", partitionKeys=[], enableUpdateCatalog=True, transformation_ctx="AmazonS3_node1745296866666")
AmazonS3_node1745296866666.setCatalogInfo(catalogDatabase="stedi_db",catalogTableName="customer_trusted")
AmazonS3_node1745296866666.setFormat("json")
AmazonS3_node1745296866666.writeFrame(SQLQuery_node1745296856903)
job.commit()