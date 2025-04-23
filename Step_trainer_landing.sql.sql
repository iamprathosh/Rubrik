CREATE TABLE "mytable" (
  "CREATE EXTERNAL TABLE `step_trainer_landing`(" text
);

INSERT INTO "mytable" ("CREATE EXTERNAL TABLE `step_trainer_landing`(")
VALUES
('`sensorreadingtime` bigint COMMENT ''from deserializer'''),
('`serialnumber` string COMMENT ''from deserializer'''),
('`distancefromobject` int COMMENT ''from deserializer'')'),
('ROW FORMAT SERDE'),
('''org.openx.data.jsonserde.JsonSerDe'''),
('STORED AS INPUTFORMAT'),
('''org.apache.hadoop.mapred.TextInputFormat'''),
('OUTPUTFORMAT'),
('''org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'''),
('LOCATION'),
('''s3://stedi-human-sr/step_trainer/landing/'''),
('TBLPROPERTIES ('),
('''classification''=''json'')');
