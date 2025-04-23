CREATE TABLE "mytable" (
  "CREATE EXTERNAL TABLE `customer_landing`(" text
);

INSERT INTO "mytable" ("CREATE EXTERNAL TABLE `customer_landing`(")
VALUES
('`customername` string COMMENT ''from deserializer'''),
('`email` string COMMENT ''from deserializer'''),
('`phone` string COMMENT ''from deserializer'''),
('`birthday` string COMMENT ''from deserializer'''),
('`serialnumber` string COMMENT ''from deserializer'''),
('`registrationdate` bigint COMMENT ''from deserializer'''),
('`lastupdatedate` bigint COMMENT ''from deserializer'''),
('`sharewithresearchasofdate` bigint COMMENT ''from deserializer'''),
('`sharewithpublicasofdate` bigint COMMENT ''from deserializer'''),
('`sharewithfriendsasofdate` bigint COMMENT ''from deserializer'')'),
('ROW FORMAT SERDE'),
('''org.openx.data.jsonserde.JsonSerDe'''),
('STORED AS INPUTFORMAT'),
('''org.apache.hadoop.mapred.TextInputFormat'''),
('OUTPUTFORMAT'),
('''org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'''),
('LOCATION'),
('''s3://stedi-human-sr/customer/landing/'''),
('TBLPROPERTIES ('),
('''classification''=''json'')');
