CREATE TABLE select_patient_new AS
WITH stg1 AS (
 SELECT *
FROM chartevents cha
WHERE icustay_id IN (
  SELECT cha.icustay_id
  FROM icustays
)
AND itemid=640
AND cha.value IN ('Extubated','Intubated')
),stg2 AS (
 SELECT stg1.subject_id,stg1.hadm_id,stg1.icustay_id,count(*) AS num
FROM stg1
GROUP BY stg1.subject_id,stg1.hadm_id,stg1.icustay_id
),stg3 AS (
 SELECT new.*,stg2.num AS num
FROM temp_pat new
LEFT JOIN stg2
  ON stg2.icustay_id=new.icustay_id
)
 SELECT *
FROM stg3


