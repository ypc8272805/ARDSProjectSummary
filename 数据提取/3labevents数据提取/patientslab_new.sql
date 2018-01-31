/*
Author:ypc
Date:2018.01.29
Table:select_patient_1222 pf  labevents
Description:
本视图的目的就是根据之前患者选择情况，提取患者在chartevents中的相关信息。
提取了所有的Pao2,还有FiO2参数，但是这个表里的FiO2不是很全，只提取部分的，但是这里的最准确。
后期还会从chartevents中提取，到时候两部分合并
患者进入ICU前六小时和后7天的数据
 */
DROP MATERIALIZED VIEW IF EXISTS patientslab CASCADE;
CREATE MATERIALIZED VIEW patientslab AS
  WITH allpatients AS (
      SELECT
        select_patient.subject_id,
        select_patient.hadm_id,
        select_patient.icustay_id,
        select_patient.intime,
        select_patient.outtime,
        select_patient.exclusion_age,
        select_patient.exclusion_chest,
        select_patient.exclusion_first_stay,
        select_patient.exclusion_los,
        select_patient.exclusion_vent,
        pf.exclusion_pf
      FROM (mimiciii.select_patient
        LEFT JOIN mimiciii.pf_new ON ((select_patient.icustay_id = pf.icustay_id)))
  ), my_patients AS (
      SELECT
        allpatients.subject_id,
        allpatients.hadm_id,
        allpatients.icustay_id,
        allpatients.intime,
        allpatients.outtime,
        allpatients.exclusion_age,
        allpatients.exclusion_chest,
        allpatients.exclusion_first_stay,
        allpatients.exclusion_los,
        allpatients.exclusion_vent,
        allpatients.exclusion_pf
      FROM allpatients
      WHERE ((allpatients.exclusion_pf = 0) AND (allpatients.exclusion_vent = 0) AND (allpatients.exclusion_los = 0) AND
             (allpatients.exclusion_first_stay = 0) AND (allpatients.exclusion_chest = 0) AND
             (allpatients.exclusion_age = 0))
  ), pvt AS
  ( -- begin query that extracts the data
      SELECT
        ie.subject_id,
        ie.hadm_id,
        ie.icustay_id
        -- here we assign labels to ITEMIDs
        -- this also fuses together multiple ITEMIDs containing the same data
        ,
        CASE
        WHEN itemid = 50819
          THEN 'PEEP'
        WHEN itemid = 50816
          THEN 'FIO2'
        WHEN itemid = 50821
          THEN 'PO2'
        ELSE NULL
        END AS label,
        charttime,
        value
        -- add in some sanity checks on the values
        ,
        CASE
        WHEN valuenum <= 0
          THEN NULL
        WHEN itemid = 50819 AND valuenum > 50
          THEN NULL --PEEP
        WHEN itemid = 50821 AND valuenum > 800
          THEN NULL -- PO2
        WHEN itemid = 50816 AND valuenum > 100 OR valuenum < 21
          THEN NULL -- FiO2
        -- conservative upper limit
        ELSE valuenum
        END AS valuenum
      FROM my_patients ie
        LEFT JOIN labevents le
          ON le.subject_id = ie.subject_id AND le.hadm_id = ie.hadm_id
             AND le.charttime BETWEEN ie.intime AND ie.outtime
             AND le.ITEMID IN (50819, 50816, 50821)
  )
  SELECT
    pvt.SUBJECT_ID,
    pvt.HADM_ID,
    pvt.ICUSTAY_ID,
    pvt.CHARTTIME,
    max(CASE WHEN label = 'PEEP'
      THEN valuenum
        ELSE NULL END) AS PEEP,
    max(CASE WHEN label = 'FIO2'
      THEN valuenum
        ELSE NULL END) AS FIO2,
    max(CASE WHEN label = 'PO2'
      THEN valuenum
        ELSE NULL END) AS PO2
  FROM pvt
  GROUP BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.CHARTTIME
  ORDER BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.CHARTTIME
