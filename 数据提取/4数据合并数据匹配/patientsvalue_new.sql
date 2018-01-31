/*
Author:ypc
Date:2018.01.29
Table:
Description:
由于之前患者的生理参数分布在两个视图：patientlab  patientchart 现在要讲两个表合在一起，所以才有了这个视图，当然这都是过
度的产物，后面都不会再使用了，这个表没有什么需要注意的，非常简单。
 */
CREATE MATERIALIZED VIEW patientvalue_1222 AS WITH allvalue AS (
    SELECT
      cha.subject_id AS subject_id_c,
      cha.hadm_id    AS hadm_id_c,
      cha.icustay_id AS icustay_id_c,
      cha.charttime  AS charttime_c,
      cha.spo2       AS spo2,
      cha.pao2       AS pao2_c,
      cha.fio2       AS fio2_c,
      cha.hr         AS hr,
      cha.temp       AS temp,
      cha.nbps       AS nbps,
      cha.nbpd       AS nbpd,
      cha.nbpm       AS nbpm,
      cha.abps       AS abps,
      cha.abpd       AS abpd,
      cha.abpm       AS abpm,
      cha.rr         AS rr,
      cha.tv         AS tv,
      cha.pip        AS pip,
      cha.plap       AS plap,
      cha.mv         AS mv,
      cha.map        AS map,
      cha.peep       AS peep_c,
      lab.subject_id AS subject_id_l,
      lab.hadm_id    AS hadm_id_l,
      lab.icustay_id AS icustay_id_l,
      lab.charttime  AS charttime_l,
      lab.po2        AS pao2_l,
      lab.fio2       AS fio2_l,
      lab.peep       AS peep_l
    FROM patientchart_1222 cha
      FULL JOIN patientslab_1222 lab
        ON cha.icustay_id = lab.icustay_id AND cha.charttime = lab.charttime
)
SELECT
  coalesce(subject_id_c, subject_id_l) AS subject_id,
  coalesce(hadm_id_c, hadm_id_l)       AS hadm_id,
  coalesce(icustay_id_c, icustay_id_l) AS icustay_id,
  coalesce(charttime_c, charttime_l)   AS charttime,
  spo2,
  coalesce(pao2_l, pao2_c)             AS pao2,
  pao2_c,
  pao2_l,
  coalesce(fio2_l, fio2_c)             AS fio2,
  fio2_c,
  fio2_l,
  hr,
  temp,
  nbps,
  nbpd,
  nbpm,
  abps,
  abpd,
  abpm,
  rr,
  tv,
  pip,
  plap,
  mv,
  map,
  coalesce(peep_l, peep_c)             AS peep,
  peep_c,
  peep_l
FROM allvalueCREATE TABLE patientsvalue_new AS
WITH stg1 AS (
    SELECT *
    FROM chartevents cha
    WHERE icustay_id IN (
      SELECT cha.icustay_id
      FROM icustays
    )
          AND itemid = 640
          AND cha.value IN ('Extubated', 'Intubated')
), stg2 AS (
    SELECT
      stg1.subject_id,
      stg1.hadm_id,
      stg1.icustay_id,
      count(*) AS num
    FROM stg1
    GROUP BY stg1.subject_id, stg1.hadm_id, stg1.icustay_id
), stg3 AS (
    SELECT
      new.*,
      stg2.num AS num
    FROM select_patient_new new
      LEFT JOIN stg2
        ON stg2.icustay_id = new.icustay_id
), stg4 AS (
    SELECT *
    FROM stg3
    WHERE stg3.exclusion_los = 0
          AND stg3.exclusion_age = 0
          AND stg3.exclusion_first_stay = 0
          AND stg3.exclusion_vent = 0
          AND stg3.exclusion_chest = 0
          AND stg3.num IS NOT NULL

)
SELECT patientsvalue.*
FROM stg4
  LEFT JOIN patientsvalue
    ON stg4.icustay_id = patientsvalue.icustay_id