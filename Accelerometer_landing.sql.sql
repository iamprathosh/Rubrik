CREATE TABLE "mytable" (
  "CREATE EXTERNAL TABLE `accelerometer_landing`(" text
);

INSERT INTO "mytable" ("CREATE EXTERNAL TABLE `accelerometer_landing`(")
VALUES
('`user` string COMMENT ''from deserializer'''),
('`timestamp` bigint COMMENT ''from deserializer'''),
('`x` float COMMENT ''from deserializer'''),
('`y` float COMMENT ''from deserializer'''),
('`z` float COMMENT ''from deserializer'')'),
('ROW FORMAT SERDE'),
('''org.openx.data.jsonserde.JsonSerDe'''),
('STORED AS INPUTFORMAT'),
('''org.apache.hadoop.mapred.TextInputFormat'''),
('OUTPUTFORMAT'),
('''org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'''),
('LOCATION'),
('''s3://stedi-human-sr/accelerometer/landing/'''),
('TBLPROPERTIES ('),
('''classification''=''json'')');
