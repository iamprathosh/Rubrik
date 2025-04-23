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

# Script generated for node Step_trainer_landing
Step_trainer_landing_node1745302629180 = glueContext.create_dynamic_frame.from_catalog(database="stedi_db", table_name="step_trainer_landing", transformation_ctx="Step_trainer_landing_node1745302629180")

# Script generated for node customer_curated
customer_curated_node1745302646044 = glueContext.create_dynamic_frame.from_catalog(database="stedi_db", table_name="customer_curated", transformation_ctx="customer_curated_node1745302646044")

# Script generated for node SQL Query
SqlQuery657 = '''
select Step_trainer.* from Step_trainer 
inner join Customer_curated
on Step_Trainer.serialnumber = Customer_curated.serialnumber;

'''
SQLQuery_node1745304303505 = sparkSqlQuery(glueContext, query = SqlQuery657, mapping = {"Step_trainer":Step_trainer_landing_node1745302629180, "Customer_curated":customer_curated_node1745302646044}, transformation_ctx = "SQLQuery_node1745304303505")

# Script generated for node Amazon S3
EvaluateDataQuality().process_rows(frame=SQLQuery_node1745304303505, ruleset=DEFAULT_DATA_QUALITY_RULESET, publishing_options={"dataQualityEvaluationContext": "EvaluateDataQuality_node1745302609004", "enableDataQualityResultsPublishing": True}, additional_options={"dataQualityResultsPublishing.strategy": "BEST_EFFORT", "observations.scope": "ALL"})
AmazonS3_node1745304953092 = glueContext.getSink(path="s3://stedi-human-sr/step_trainer/trusted/", connection_type="s3", updateBehavior="UPDATE_IN_DATABASE", partitionKeys=[], enableUpdateCatalog=True, transformation_ctx="AmazonS3_node1745304953092")
AmazonS3_node1745304953092.setCatalogInfo(catalogDatabase="stedi_db",catalogTableName="step_trainer_trusted")
AmazonS3_node1745304953092.setFormat("json")
AmazonS3_node1745304953092.writeFrame(SQLQuery_node1745304303505)
job.commit()