/*
Author:ypc
Date:2018.01.29
Table:select_patient_1222 pf  chartevents
Description:
本视图的目的就是根据之前患者选择情况，提取患者在chartevents中的相关信息。
 */
DROP MATERIALIZED VIEW IF EXISTS patientchart_1222 CASCADE;
CREATE MATERIALIZED VIEW patientchart_1222 AS
  --这段内容是由于在一开始选择患者的时候是分布在两个视图中的，一个视图标注了一些条件，但是pf值得约束在pf这个表中，所以要将
  -- 两个表关联
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
      FROM select_patient_1222 select_patient
        LEFT JOIN pf
          ON select_patient.icustay_id = pf.icustay_id
  ), my_patients AS (
    --得到了最终的患者
      SELECT *
      FROM allpatients
      WHERE exclusion_pf = 0 AND exclusion_vent = 0 AND exclusion_los = 0 AND exclusion_first_stay = 0 AND
            exclusion_chest = 0 AND exclusion_age = 0
  ), chart_value AS (
    --提取患者的生理参数
      SELECT
        ie.subject_id,
        ie.hadm_id,
        ie.icustay_id,
        CASE
        WHEN itemid IN (220277, 646)
          THEN 'SPO2'
        WHEN itemid IN (220224, 779)
          THEN 'PAO2'
        WHEN itemid IN (190, 3420, 223835, 3422)
          THEN 'FIO2'
        WHEN itemid IN (211, 220045)
          THEN 'HR'
        WHEN itemid IN (223762, 676, 677, 223761, 678, 679)
          THEN 'TEMP'
        WHEN itemid IN (442, 455, 220179)
          THEN 'NBPS'
        WHEN itemid IN (8440, 8441, 220180)
          THEN 'NBPD'
        WHEN itemid IN (456, 443, 220181)
          THEN 'NBPM'
        WHEN itemid IN (51, 6701, 220050)
          THEN 'ABPS'
        WHEN itemid IN (8368, 8555, 220051)
          THEN 'ABPD'
        WHEN itemid IN (52, 220052, 6702, 225312)
          THEN 'ABPM'
        WHEN itemid IN (615, 618, 220210, 224690)
          THEN 'RR'
        WHEN itemid IN (681, 682, 2400, 2408, 2534, 2420, 224685)
          THEN 'TV'
        WHEN itemid IN (507, 535, 224695)
          THEN 'PIP'
        WHEN itemid IN (543, 224696)
          THEN 'PLAP'
        WHEN itemid IN (445, 448, 450, 224687)
          THEN 'MV'
        WHEN itemid IN (444, 3502, 3503, 224697)
          THEN 'MAP'
        WHEN itemid IN (506, 220339)
          THEN 'PEEP'
        ELSE NULL
        END AS label,
        intime,
        outtime,
        charttime,
        itemid,
        CASE
        WHEN itemid IN (220277, 646) AND valuenum > 0 AND valuenum <= 100
          THEN valuenum -- spo2
        WHEN itemid IN (220224, 779) AND valuenum > 0 AND valuenum < 800
          THEN valuenum -- pao2
        WHEN itemid IN (211, 220045) AND valuenum > 0 AND valuenum < 300
          THEN valuenum --hr
        WHEN itemid = 223835 --fio2
          THEN CASE
               WHEN valuenum > 0 AND valuenum <= 1
                 THEN valuenum * 100
               WHEN valuenum > 1 AND valuenum < 21
                 THEN NULL
               WHEN valuenum >= 21 AND valuenum <= 100
                 THEN valuenum
               ELSE NULL END
        WHEN itemid IN (3420, 3422) AND valuenum > 0 AND valuenum < 100
          THEN valuenum
        WHEN itemid = 190 AND valuenum > 0.2 AND valuenum <= 1
          THEN valuenum * 100
        WHEN itemid IN (223762, 676, 677) AND valuenum > 10 AND valuenum < 50
          THEN valuenum -- celsius
        WHEN itemid IN (223761, 678, 679) AND valuenum > 70 AND valuenum < 120
          THEN (valuenum - 32) * 5 / 9 --fahrenheit
        WHEN itemid IN (51, 442, 455, 6701, 220179, 220050) AND valuenum > 0 AND valuenum < 400
          THEN valuenum --bps
        WHEN itemid IN (8368, 8440, 8441, 8555, 220180, 220051) AND valuenum > 0 AND valuenum < 300
          THEN valuenum --bpd
        WHEN itemid IN (456, 52, 6702, 443, 220052, 220181, 225312) AND valuenum > 0 AND valuenum < 300
          THEN valuenum --bpm
        WHEN itemid IN (615, 618, 220210, 224690) AND valuenum >= 0 AND valuenum < 70
          THEN valuenum --rr
        WHEN itemid IN (681, 682, 2400, 2408, 2534, 2420, 224685) AND valuenum > 100 AND valuenum < 2000
          THEN valuenum --tv
        WHEN itemid IN (507, 535, 224695) AND valuenum > 0 AND valuenum < 100
          THEN valuenum --pip
        WHEN itemid IN (543, 224696) AND valuenum > 0 AND valuenum < 100
          THEN valuenum --plap
        WHEN itemid IN (445, 448, 450, 224687) AND valuenum > 0 AND valuenum < 50
          THEN valuenum --mv
        WHEN itemid IN (444, 3502, 3503, 224697) AND valuenum > 0 AND valuenum < 100
          THEN valuenum --map
        WHEN itemid IN (506, 220339) AND valuenum > 0 AND valuenum < 40
          THEN valuenum --peep
        ELSE NULL
        END AS valuenum,
        valueuom
      FROM my_patients ie
        LEFT JOIN chartevents le
          ON le.subject_id = ie.subject_id AND le.hadm_id = ie.hadm_id AND ie.icustay_id = le.icustay_id
             AND le.charttime BETWEEN ie.intime AND ie.outtime
             AND le.ITEMID IN
                 -- 我需要的参数
                 (
                   220277, 646--spo2
                   , 220224, 779--pao2
                   , 190, 3420, 3422, 223835--fio2
                   , 211, 220045--hr
                   , 223762, 676, 677, 223761, 678, 679--temp
                   , 442, 455, 220179--nbps
                   , 8440, 8441, 220180--nbpd
                   , 456, 443, 220181--nbpm
                   , 51, 6701, 220050--abps
                   , 8368, 8555, 220051--abpd
                   , 52, 220052, 6702, 225312--abpm
                   , 615, 618, 220210, 224690--rr
                   , 681, 682, 2400, 2408, 2534, 2420, 224685--tv
                   , 507, 535, 224695--pip
                   , 543, 224696--plap
                   , 445, 448, 450, 224687--mv
                   , 444, 3502, 3503, 224697--map
                   , 506, 220339--peep
                 )
  )
  --由于提取的数据是按照时间顺序排列的，后期不好处理，在这里做了一个透视图，行是时间列是参数
  SELECT
    pvt.SUBJECT_ID,
    pvt.HADM_ID,
    pvt.ICUSTAY_ID,
    pvt.CHARTTIME,
    max(CASE WHEN label = 'SPO2'
      THEN valuenum
        ELSE NULL END) AS SPO2,
    max(CASE WHEN label = 'PAO2'
      THEN valuenum
        ELSE NULL END) AS PAO2,
    max(CASE WHEN label = 'FIO2'
      THEN valuenum
        ELSE NULL END) AS FIO2,
    max(CASE WHEN label = 'HR'
      THEN valuenum
        ELSE NULL END) AS HR,
    max(CASE WHEN label = 'TEMP'
      THEN valuenum
        ELSE NULL END) AS TEMP,
    max(CASE WHEN label = 'NBPS'
      THEN valuenum
        ELSE NULL END) AS NBPS,
    max(CASE WHEN label = 'NBPD'
      THEN valuenum
        ELSE NULL END) AS NBPD,
    max(CASE WHEN label = 'NBPM'
      THEN valuenum
        ELSE NULL END) AS NBPM,
    max(CASE WHEN label = 'ABPS'
      THEN valuenum
        ELSE NULL END) AS ABPS,
    max(CASE WHEN label = 'ABPD'
      THEN valuenum
        ELSE NULL END) AS ABPD,
    max(CASE WHEN label = 'ABPM'
      THEN valuenum
        ELSE NULL END) AS ABPM,
    max(CASE WHEN label = 'RR'
      THEN valuenum
        ELSE NULL END) AS RR,
    max(CASE WHEN label = 'TV'
      THEN valuenum
        ELSE NULL END) AS TV,
    max(CASE WHEN label = 'PIP'
      THEN valuenum
        ELSE NULL END) AS PIP,
    max(CASE WHEN label = 'PLAP'
      THEN valuenum
        ELSE NULL END) AS PLAP,
    max(CASE WHEN label = 'MV'
      THEN valuenum
        ELSE NULL END) AS MV,
    max(CASE WHEN label = 'MAP'
      THEN valuenum
        ELSE NULL END) AS MAP,
    max(CASE WHEN label = 'PEEP'
      THEN valuenum
        ELSE NULL END) AS PEEP
  FROM chart_value pvt
  GROUP BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.CHARTTIME
  ORDER BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.CHARTTIME