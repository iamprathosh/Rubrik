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

# Script generated for node accelerometer_trusted
accelerometer_trusted_node1745309039074 = glueContext.create_dynamic_frame.from_catalog(database="stedi_db", table_name="accelerometer_trusted", transformation_ctx="accelerometer_trusted_node1745309039074")

# Script generated for node step_trainer_trusted
step_trainer_trusted_node1745309037596 = glueContext.create_dynamic_frame.from_catalog(database="stedi_db", table_name="step_trainer_trusted", transformation_ctx="step_trainer_trusted_node1745309037596")

# Script generated for node SQL Query
SqlQuery639 = '''
select step_trainer_trusted.sensorreadingtime,step_trainer_trusted.serialnumber,
step_trainer_trusted.distancefromobject,accelerometer_trusted.user,accelerometer_trusted.x,
accelerometer_trusted.y,accelerometer_trusted.z
from step_trainer_trusted 
inner join accelerometer_trusted
on accelerometer_trusted.timestamp=step_trainer_trusted.sensorreadingtime;


'''
SQLQuery_node1745309045178 = sparkSqlQuery(glueContext, query = SqlQuery639, mapping = {"accelerometer_trusted":accelerometer_trusted_node1745309039074, "step_trainer_trusted":step_trainer_trusted_node1745309037596}, transformation_ctx = "SQLQuery_node1745309045178")

# Script generated for node Amazon S3
EvaluateDataQuality().process_rows(frame=SQLQuery_node1745309045178, ruleset=DEFAULT_DATA_QUALITY_RULESET, publishing_options={"dataQualityEvaluationContext": "EvaluateDataQuality_node1745309024333", "enableDataQualityResultsPublishing": True}, additional_options={"dataQualityResultsPublishing.strategy": "BEST_EFFORT", "observations.scope": "ALL"})
AmazonS3_node1745309050339 = glueContext.getSink(path="s3://stedi-human-sr/step_trainer/curated/", connection_type="s3", updateBehavior="UPDATE_IN_DATABASE", partitionKeys=[], enableUpdateCatalog=True, transformation_ctx="AmazonS3_node1745309050339")
AmazonS3_node1745309050339.setCatalogInfo(catalogDatabase="stedi_db",catalogTableName="machine_learning_curated")
AmazonS3_node1745309050339.setFormat("json")
AmazonS3_node1745309050339.writeFrame(SQLQuery_node1745309045178)
job.commit()