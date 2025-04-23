import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsgluedq.transforms import EvaluateDataQuality
from awsglue import DynamicFrame

def sql_transform(context, sql_text, view_definitions, ctx_id):
    """
    Execute SQL transformation on dynamic frames.
    
    Args:
        context: GlueContext to use
        sql_text: The SQL query to execute
        view_definitions: Dictionary mapping view names to DynamicFrames
        ctx_id: Context ID for the transformation
    
    Returns:
        DynamicFrame containing query results
    """
    # Register each dynamic frame as a temp view
    for view_name, dynamic_frame in view_definitions.items():
        dynamic_frame.toDF().createOrReplaceTempView(view_name)
    
    # Run the query and convert result back to DynamicFrame
    result_dataframe = spark.sql(sql_text)
    return DynamicFrame.fromDF(result_dataframe, context, ctx_id)

# Set up the Glue job
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Define data quality validation rules
DATA_QUALITY_RULES = """
    Rules = [
        ColumnCount > 0
    ]
"""

# Load accelerometer readings from trusted zone
# These readings are already filtered to only include data from consenting users
accelerometer_trusted = glueContext.create_dynamic_frame.from_catalog(
    database="stedi_db", 
    table_name="accelerometer_trusted", 
    transformation_ctx="load_accelerometer_trusted"
)

# Load step trainer readings from trusted zone
# These readings are already filtered to only include data from consenting users' devices
step_trainer_trusted = glueContext.create_dynamic_frame.from_catalog(
    database="stedi_db", 
    table_name="step_trainer_trusted", 
    transformation_ctx="load_step_trainer_trusted"
)

# Join accelerometer and step trainer data based on timestamp
# This creates a machine learning dataset that links customer movement with step trainer readings
ml_dataset_query = """
SELECT 
    step_data.sensorreadingtime, 
    step_data.serialnumber,
    step_data.distancefromobject, 
    accel_data.user, 
    accel_data.x,
    accel_data.y, 
    accel_data.z
FROM step_data
INNER JOIN accel_data
ON accel_data.timestamp = step_data.sensorreadingtime
"""

ml_curated_dataset = sql_transform(
    glueContext,
    ml_dataset_query,
    {"accel_data": accelerometer_trusted, "step_data": step_trainer_trusted},
    "create_ml_dataset"
)

# Validate the quality of the final dataset
EvaluateDataQuality().process_rows(
    frame=ml_curated_dataset, 
    ruleset=DATA_QUALITY_RULES, 
    publishing_options={
        "dataQualityEvaluationContext": "ml_dataset_quality_validation", 
        "enableDataQualityResultsPublishing": True
    }, 
    additional_options={
        "dataQualityResultsPublishing.strategy": "BEST_EFFORT", 
        "observations.scope": "ALL"
    }
)

# Write the machine learning curated dataset to S3
ml_dataset_sink = glueContext.getSink(
    path="s3://stedi-human-sr/step_trainer/curated/", 
    connection_type="s3", 
    updateBehavior="UPDATE_IN_DATABASE", 
    partitionKeys=[], 
    enableUpdateCatalog=True, 
    transformation_ctx="write_ml_curated_dataset"
)

ml_dataset_sink.setCatalogInfo(
    catalogDatabase="stedi_db", 
    catalogTableName="machine_learning_curated"
)
ml_dataset_sink.setFormat("json")
ml_dataset_sink.writeFrame(ml_curated_dataset)

job.commit()