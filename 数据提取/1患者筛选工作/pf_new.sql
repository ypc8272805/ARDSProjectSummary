/*
Author:ypc
Date:2018.01.29
Table:Chartevents pao2oneday
Description:
这个视图是筛选患者的一部分，对应的条件就是患者进入ICU后第一天的P/F值出现小于300的情况，就把病人标记为1，将这一部分的结果
与select patient脚本结果结合，实现患者的筛选。你也会发现，pf view 有两个视图，这是由于，在一开始制定筛选条件的时候，我们
将条件限定为患者进入ICU7天以内，但是后来发现，这样定义太宽泛，会纳入很多混杂因素，实验结果不理想，鉴于此，在后期，我们改
进了标准，这些改进不仅仅是对pf view的改进，还对select patient进行了改进。

这个视图，主要是通过chartevents,pao2oneday来计算患者的P/F值，从而判断患者第一天平均的P/F是否小于300，用于定义
ARDS。最终输出的结果，是基于icustays表，最后加了一列，1--P/F>300 ;0--P/F<=300。
 */
CREATE TABLE pf_new AS
  WITH stg_fio2 AS (
      SELECT
        chartevents.subject_id,
        chartevents.hadm_id,
        chartevents.icustay_id,
        chartevents.charttime,
        --提取患者的FiO2参数
        --Extract the patient's physiological parameters FiO2
        max(
            CASE
            WHEN (chartevents.itemid = 223835)
              THEN
                CASE
                WHEN ((chartevents.valuenum > (0) :: DOUBLE PRECISION) AND
                      (chartevents.valuenum <= (1) :: DOUBLE PRECISION))
                  THEN (chartevents.valuenum * (100) :: DOUBLE PRECISION)
                WHEN ((chartevents.valuenum > (1) :: DOUBLE PRECISION) AND
                      (chartevents.valuenum < (21) :: DOUBLE PRECISION))
                  THEN NULL :: DOUBLE PRECISION
                WHEN ((chartevents.valuenum >= (21) :: DOUBLE PRECISION) AND
                      (chartevents.valuenum <= (100) :: DOUBLE PRECISION))
                  THEN chartevents.valuenum
                ELSE NULL :: DOUBLE PRECISION
                END
            WHEN (chartevents.itemid = ANY (ARRAY [3420, 3422]))
              THEN chartevents.valuenum
            WHEN ((chartevents.itemid = 190) AND (chartevents.valuenum > (0.20) :: DOUBLE PRECISION) AND
                  (chartevents.valuenum < (1) :: DOUBLE PRECISION))
              THEN (chartevents.valuenum * (100) :: DOUBLE PRECISION)
            ELSE NULL :: DOUBLE PRECISION
            END) AS fio2_chartevents
      FROM mimiciii.chartevents
      WHERE ((chartevents.itemid = ANY (ARRAY [3420, 190, 223835, 3422])) AND (chartevents.error IS DISTINCT FROM 1))
      GROUP BY chartevents.subject_id, chartevents.hadm_id, chartevents.icustay_id, chartevents.charttime
  ), stg1 AS (
    --在上面的基础上，关联患者时间范围内的所有FiO2值。
    --Based on the above, all FiO2 values in the patient's time range are correlated.
      SELECT
        po2.subject_id,
        po2.hadm_id,
        po2.icustay_id,
        po2.charttime,
        po2.o2flow,
        po2.fio2,
        po2.po2,
        fio2.fio2_chartevents,
        row_number()
        OVER (
          PARTITION BY po2.icustay_id, po2.charttime
          ORDER BY fio2.charttime DESC ) AS lastrowfio2
      FROM (mimiciii.po2oneday po2
        LEFT JOIN stg_fio2 fio2 ON (((po2.icustay_id = fio2.icustay_id) AND
                                     ((fio2.charttime >= (po2.charttime - '04:00:00' :: INTERVAL HOUR)) AND
                                      (fio2.charttime <= po2.charttime)))))
  ), stg2 AS (
    --只选择PaO2附近了FiO2值，来计算P/F
    --Only FiO2 values are selected near PaO2 to calculate P / F
      SELECT
        stg1.subject_id,
        stg1.hadm_id,
        stg1.icustay_id,
        stg1.charttime,
        stg1.o2flow,
        stg1.fio2,
        stg1.po2,
        stg1.fio2_chartevents,
        stg1.lastrowfio2,
        CASE
        WHEN ((stg1.po2 IS NOT NULL) AND (COALESCE(stg1.fio2, stg1.fio2_chartevents) IS NOT NULL))
          THEN (((100) :: DOUBLE PRECISION * stg1.po2) / COALESCE(stg1.fio2, stg1.fio2_chartevents))
        ELSE NULL :: DOUBLE PRECISION
        END AS pf
      FROM stg1
      WHERE (stg1.lastrowfio2 = 1)
      ORDER BY stg1.icustay_id, stg1.charttime
  ), stg3 AS (
    --计算患者第一天的平均P/F值
    --Calculate the average P / F value of the first day of the patient
      SELECT
        stg2.subject_id,
        stg2.hadm_id,
        stg2.icustay_id,
        avg(stg2.pf) AS avgpf
      FROM stg2
      GROUP BY stg2.subject_id, stg2.hadm_id, stg2.icustay_id
  )
  --根据P/F对患者进行标注
  --Patients were labeled according to P / F
  SELECT
    stg3.subject_id,
    stg3.hadm_id,
    stg3.icustay_id,
    stg3.avgpf,
    CASE
    WHEN ((stg3.avgpf > (300) :: DOUBLE PRECISION) OR (stg3.avgpf IS NULL))
      THEN 1
    ELSE 0
    END AS exclusion_pf
  FROM stg3;
