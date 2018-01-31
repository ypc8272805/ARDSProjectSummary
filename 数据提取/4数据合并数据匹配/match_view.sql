/*
Author:ypc
Date:2018.01.29
Table:
Description:
本视图的目的就是对之前的patientvalue表做一个透视图，方便后续分析使用，在patientvalue中，所有变量都是单独存放的，现在要按
照pao2对所有参数进行匹配，以前这部分工作使用matlab进行分析，非常繁琐，现在使用SQL非常方便，不同参数的时间误差在论文中有
明确的定义。
 */
CREATE MATERIALIZED VIEW matchvalue_1222 AS
  --首先是将所有参数从patientvalue中提取出来，每一个参数创建一个临时表，方便后面使用，但是这样写很繁琐，应该有更好的方法，
  -- 但是目前我还没有想到。
  WITH pao2 AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        pao2
      FROM patientvalue_1222
      WHERE pao2 IS NOT NULL
  ), spo2 AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        spo2
      FROM patientvalue_1222
      WHERE patientvalue_1222.spo2 IS NOT NULL
  ), fio2 AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        fio2
      FROM patientvalue_1222
      WHERE fio2 IS NOT NULL
  ), hr AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        hr
      FROM patientvalue_1222
      WHERE hr IS NOT NULL
  ), temp AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        temp
      FROM patientvalue_1222
      WHERE patientvalue_1222.temp IS NOT NULL
  ), nbps AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        nbps
      FROM patientvalue_1222
      WHERE patientvalue_1222.nbps IS NOT NULL
  ), nbpd AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        nbpd
      FROM patientvalue_1222
      WHERE patientvalue_1222.nbpd IS NOT NULL
  ), nbpm AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        nbpm
      FROM patientvalue_1222
      WHERE patientvalue_1222.nbpm IS NOT NULL
  ), abps AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        abps
      FROM patientvalue_1222
      WHERE patientvalue_1222.abps IS NOT NULL
  ), abpd AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        abpd
      FROM patientvalue_1222
      WHERE patientvalue_1222.abpd IS NOT NULL
  ), abpm AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        abpm
      FROM patientvalue_1222
      WHERE patientvalue_1222.abpm IS NOT NULL
  ), rr AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        rr
      FROM patientvalue_1222
      WHERE patientvalue_1222.rr IS NOT NULL
  ), tv AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        tv
      FROM patientvalue_1222
      WHERE patientvalue_1222.tv IS NOT NULL
  ), pip AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        pip
      FROM patientvalue_1222
      WHERE patientvalue_1222.pip IS NOT NULL
  ), plap AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        plap
      FROM patientvalue_1222
      WHERE patientvalue_1222.plap IS NOT NULL
  ), mv AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        mv
      FROM patientvalue_1222
      WHERE patientvalue_1222.mv IS NOT NULL
  ), map AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        map
      FROM patientvalue_1222
      WHERE patientvalue_1222.map IS NOT NULL
  ), peep AS (
      SELECT
        subject_id,
        hadm_id,
        icustay_id,
        charttime,
        peep
      FROM patientvalue_1222
      WHERE patientvalue_1222.peep IS NOT NULL
  ), patvalue AS (
    --以pao2为标准，匹配2小时以前的spo2数据，如果没有就为空，如果有多条，选择最近的一条，也就是选择lastRowSpO2=1的就OK了
      SELECT
        pao2.subject_id,
        pao2.hadm_id,
        pao2.icustay_id,
        pao2.charttime,
        pao2.pao2,
        spo2.spo2,
        spo2.charttime                           AS s_time,
        CASE WHEN pao2.charttime > spo2.charttime
          THEN pao2.charttime - spo2.charttime
        ELSE spo2.charttime - pao2.charttime END AS sc
      FROM pao2
        LEFT JOIN spo2
          ON pao2.icustay_id = spo2.icustay_id
             AND (spo2.charttime BETWEEN pao2.charttime - INTERVAL '2' HOUR AND pao2.charttime
                  OR spo2.charttime BETWEEN pao2.charttime AND pao2.charttime + INTERVAL '2' HOUR)
  ), patvalue_1 AS (
      SELECT
        *,
        row_number()
        OVER (
          PARTITION BY icustay_id, charttime
          ORDER BY sc ) AS lastRowSpO2
      FROM patvalue
  ), stg1 AS (
      SELECT
        pat.subject_id,
        pat.hadm_id,
        pat.icustay_id,
        pat.charttime,
        pat.pao2,
        pat.spo2,
        row_number()
        OVER (
          PARTITION BY pat.icustay_id, pat.charttime
          ORDER BY fio2.charttime DESC ) AS lastRowFiO2,
        fio2.fio2
      FROM patvalue_1 pat
        LEFT JOIN fio2
          ON fio2.icustay_id = pat.icustay_id
             AND fio2.charttime BETWEEN pat.charttime - INTERVAL '12' HOUR AND pat.charttime
      WHERE pat.lastRowSpO2 = 1
  ), stg2 AS (
      SELECT
        pat.subject_id,
        pat.hadm_id,
        pat.icustay_id,
        pat.charttime,
        pat.pao2,
        pat.spo2,
        pat.fio2
        --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowHR
        ,
        CASE WHEN pat.charttime > s2.charttime
          THEN pat.charttime - s2.charttime
        ELSE s2.charttime - pat.charttime END AS sc,
        s2.hr
      FROM stg1 pat
        LEFT JOIN hr s2
          ON s2.icustay_id = pat.icustay_id
             AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                  OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
      WHERE pat.lastRowFiO2 = 1
  ), stg2_1 AS (
      SELECT
        *,
        row_number()
        OVER (
          PARTITION BY icustay_id, charttime
          ORDER BY sc ) AS lastRowHR
      FROM stg2
  ), stg3 AS (
      SELECT
        pat.subject_id,
        pat.hadm_id,
        pat.icustay_id,
        pat.charttime,
        pat.pao2,
        pat.spo2,
        pat.fio2,
        pat.hr
        --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowTemp
        ,
        CASE WHEN pat.charttime > s2.charttime
          THEN pat.charttime - s2.charttime
        ELSE s2.charttime - pat.charttime END AS sc,
        s2.temp
      FROM stg2_1 pat
        LEFT JOIN temp s2
          ON s2.icustay_id = pat.icustay_id
             AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                  OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
      WHERE pat.lastRowHR = 1
  ), stg3_1 AS (
      SELECT
        *,
        row_number()
        OVER (
          PARTITION BY icustay_id, charttime
          ORDER BY sc ) AS lastRowTemp
      FROM stg3
  ), stg4 AS (
      SELECT
        pat.subject_id,
        pat.hadm_id,
        pat.icustay_id,
        pat.charttime,
        pat.pao2,
        pat.spo2,
        pat.fio2,
        pat.hr,
        pat.temp
        --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowNbps
        ,
        CASE WHEN pat.charttime > s2.charttime
          THEN pat.charttime - s2.charttime
        ELSE s2.charttime - pat.charttime END AS sc,
        s2.nbps
      FROM stg3_1 pat
        LEFT JOIN nbps s2
          ON s2.icustay_id = pat.icustay_id
             AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                  OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
      WHERE pat.lastRowTemp = 1
  ), stg4_1 AS (
      SELECT
        *,
        row_number()
        OVER (
          PARTITION BY icustay_id, charttime
          ORDER BY sc ) AS lastRowNbps
      FROM stg4
  ), stg5 AS (
      SELECT
        pat.subject_id,
        pat.hadm_id,
        pat.icustay_id,
        pat.charttime,
        pat.pao2,
        pat.spo2,
        pat.fio2,
        pat.hr,
        pat.temp,
        pat.nbps
        --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowNbpd
        ,
        CASE WHEN pat.charttime > s2.charttime
          THEN pat.charttime - s2.charttime
        ELSE s2.charttime - pat.charttime END AS sc,
        s2.nbpd
      FROM stg4_1 pat
        LEFT JOIN nbpd s2
          ON s2.icustay_id = pat.icustay_id
             AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                  OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
      WHERE pat.lastRowNbps = 1
  ), stg5_1 AS (
      SELECT
        *,
        row_number()
        OVER (
          PARTITION BY icustay_id, charttime
          ORDER BY sc ) AS lastRowNbpd
      FROM stg5
  ), stg6 AS (
      SELECT
        pat.subject_id,
        pat.hadm_id,
        pat.icustay_id,
        pat.charttime,
        pat.pao2,
        pat.spo2,
        pat.fio2,
        pat.hr,
        pat.temp,
        pat.nbps,
        pat.nbpd
        --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowNbpm
        ,
        CASE WHEN pat.charttime > s2.charttime
          THEN pat.charttime - s2.charttime
        ELSE s2.charttime - pat.charttime END AS sc,
        s2.nbpm
      FROM stg5_1 pat
        LEFT JOIN nbpm s2
          ON s2.icustay_id = pat.icustay_id
             AND (s2.charttime BETWEEN pat.charttime - INTERVAL '6' HOUR AND pat.charttime
                  OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '6' HOUR)
      WHERE pat.lastRowNbpd = 1
  ), stg6_1 AS (
      SELECT
        *,
        row_number()
        OVER (
          PARTITION BY icustay_id, charttime
          ORDER BY sc ) AS lastRowNbpm
      FROM stg6
  ), stg7 AS (
      SELECT
        pat.subject_id,
        pat.hadm_id,
        pat.icustay_id,
        pat.charttime,
        pat.pao2,
        pat.spo2,
        pat.fio2,
        pat.hr,
        pat.temp,
        pat.nbps,
        pat.nbpd,
        pat.nbpm
        --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowAbps
        ,
        CASE WHEN pat.charttime > s2.charttime
          THEN pat.charttime - s2.charttime
        ELSE s2.charttime - pat.charttime END AS sc,
        s2.abps
      FROM stg6_1 pat
        LEFT JOIN abps s2
          ON s2.icustay_id = pat.icustay_id
             AND (s2.charttime BETWEEN pat.charttime - INTERVAL '6' HOUR AND pat.charttime
                  OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '6' HOUR)
      WHERE pat.lastRowNbpm = 1
  ), stg7_1 AS (
      SELECT
        *,
        row_number()
        OVER (
          PARTITION BY icustay_id, charttime
          ORDER BY sc ) AS lastRowAbps
      FROM stg7
  ), stg8 AS (
      SELECT
        pat.subject_id,
        pat.hadm_id,
        pat.icustay_id,
        pat.charttime,
        pat.pao2,
        pat.spo2,
        pat.fio2,
        pat.hr,
        pat.temp,
        pat.nbps,
        pat.nbpd,
        pat.nbpm,
        pat.abps
        --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowAbpd
        ,
        CASE WHEN pat.charttime > s2.charttime
          THEN pat.charttime - s2.charttime
        ELSE s2.charttime - pat.charttime END AS sc,
        s2.abpd
      FROM stg7_1 pat
        LEFT JOIN abpd s2
          ON s2.icustay_id = pat.icustay_id
             AND (s2.charttime BETWEEN pat.charttime - INTERVAL '6' HOUR AND pat.charttime
                  OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '6' HOUR)
      WHERE pat.lastRowAbps = 1
  ), stg8_1 AS (
      SELECT
        *,
        row_number()
        OVER (
          PARTITION BY icustay_id, charttime
          ORDER BY sc ) AS lastRowAbpd
      FROM stg8
  )
    , stg9 AS (
      SELECT
        pat.subject_id,
        pat.hadm_id,
        pat.icustay_id,
        pat.charttime,
        pat.pao2,
        pat.spo2,
        pat.fio2,
        pat.hr,
        pat.temp,
        pat.nbps,
        pat.nbpd,
        pat.nbpm,
        pat.abps,
        pat.abpd
        --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowAbpm
        ,
        CASE WHEN pat.charttime > s2.charttime
          THEN pat.charttime - s2.charttime
        ELSE s2.charttime - pat.charttime END AS sc,
        s2.abpm
      FROM stg8_1 pat
        LEFT JOIN abpm s2
          ON s2.icustay_id = pat.icustay_id
             AND (s2.charttime BETWEEN pat.charttime - INTERVAL '6' HOUR AND pat.charttime
                  OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '6' HOUR)
      WHERE pat.lastRowAbpd = 1
  ), stg9_1 AS (
      SELECT
        *,
        row_number()
        OVER (
          PARTITION BY icustay_id, charttime
          ORDER BY sc ) AS lastRowAbpm
      FROM stg9
  ), stg10 AS (
      SELECT
        pat.subject_id,
        pat.hadm_id,
        pat.icustay_id,
        pat.charttime,
        pat.pao2,
        pat.spo2,
        pat.fio2,
        pat.hr,
        pat.temp,
        pat.nbps,
        pat.nbpd,
        pat.nbpm,
        pat.abps,
        pat.abpd,
        pat.abpm
        --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowRR
        ,
        CASE WHEN pat.charttime > s2.charttime
          THEN pat.charttime - s2.charttime
        ELSE s2.charttime - pat.charttime END AS sc,
        s2.rr
      FROM stg9_1 pat
        LEFT JOIN rr s2
          ON s2.icustay_id = pat.icustay_id
             AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                  OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
      WHERE pat.lastRowAbpm = 1
  ), stg10_1 AS (
      SELECT
        *,
        row_number()
        OVER (
          PARTITION BY icustay_id, charttime
          ORDER BY sc ) AS lastRowRR
      FROM stg10
  ), stg11 AS (
      SELECT
        pat.subject_id,
        pat.hadm_id,
        pat.icustay_id,
        pat.charttime,
        pat.pao2,
        pat.spo2,
        pat.fio2,
        pat.hr,
        pat.temp,
        pat.nbps,
        pat.nbpd,
        pat.nbpm,
        pat.abps,
        pat.abpd,
        pat.abpm,
        pat.rr
        --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowTV
        ,
        CASE WHEN pat.charttime > s2.charttime
          THEN pat.charttime - s2.charttime
        ELSE s2.charttime - pat.charttime END AS sc,
        s2.tv
      FROM stg10_1 pat
        LEFT JOIN tv s2
          ON s2.icustay_id = pat.icustay_id
             AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                  OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
      WHERE pat.lastRowRR = 1
  ), stg11_1 AS (
      SELECT
        *,
        row_number()
        OVER (
          PARTITION BY icustay_id, charttime
          ORDER BY sc ) AS lastRowTV
      FROM stg11
  ), stg12 AS (
      SELECT
        pat.subject_id,
        pat.hadm_id,
        pat.icustay_id,
        pat.charttime,
        pat.pao2,
        pat.spo2,
        pat.fio2,
        pat.hr,
        pat.temp,
        pat.nbps,
        pat.nbpd,
        pat.nbpm,
        pat.abps,
        pat.abpd,
        pat.abpm,
        pat.rr,
        pat.tv
        --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowPIP
        ,
        CASE WHEN pat.charttime > s2.charttime
          THEN pat.charttime - s2.charttime
        ELSE s2.charttime - pat.charttime END AS sc,
        s2.pip
      FROM stg11_1 pat
        LEFT JOIN pip s2
          ON s2.icustay_id = pat.icustay_id
             AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                  OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
      WHERE pat.lastRowTV = 1
  ), stg12_1 AS (
      SELECT
        *,
        row_number()
        OVER (
          PARTITION BY icustay_id, charttime
          ORDER BY sc ) AS lastRowPIP
      FROM stg12
  ), stg13 AS (
      SELECT
        pat.subject_id,
        pat.hadm_id,
        pat.icustay_id,
        pat.charttime,
        pat.pao2,
        pat.spo2,
        pat.fio2,
        pat.hr,
        pat.temp,
        pat.nbps,
        pat.nbpd,
        pat.nbpm,
        pat.abps,
        pat.abpd,
        pat.abpm,
        pat.rr,
        pat.tv,
        pat.pip
        --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowPLAP
        ,
        CASE WHEN pat.charttime > s2.charttime
          THEN pat.charttime - s2.charttime
        ELSE s2.charttime - pat.charttime END AS sc,
        s2.plap
      FROM stg12_1 pat
        LEFT JOIN plap s2
          ON s2.icustay_id = pat.icustay_id
             AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                  OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
      WHERE pat.lastRowPIP = 1
  ), stg13_1 AS (
      SELECT
        *,
        row_number()
        OVER (
          PARTITION BY icustay_id, charttime
          ORDER BY sc ) AS lastRowPLAP
      FROM stg13
  ), stg14 AS (
      SELECT
        pat.subject_id,
        pat.hadm_id,
        pat.icustay_id,
        pat.charttime,
        pat.pao2,
        pat.spo2,
        pat.fio2,
        pat.hr,
        pat.temp,
        pat.nbps,
        pat.nbpd,
        pat.nbpm,
        pat.abps,
        pat.abpd,
        pat.abpm,
        pat.rr,
        pat.tv,
        pat.pip,
        pat.plap
        --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowMV
        ,
        CASE WHEN pat.charttime > s2.charttime
          THEN pat.charttime - s2.charttime
        ELSE s2.charttime - pat.charttime END AS sc,
        s2.mv
      FROM stg13_1 pat
        LEFT JOIN mv s2
          ON s2.icustay_id = pat.icustay_id
             AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                  OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
      WHERE pat.lastRowPLAP = 1
  ), stg14_1 AS (
      SELECT
        *,
        row_number()
        OVER (
          PARTITION BY icustay_id, charttime
          ORDER BY sc ) AS lastRowMV
      FROM stg14
  ), stg15 AS (
      SELECT
        pat.subject_id,
        pat.hadm_id,
        pat.icustay_id,
        pat.charttime,
        pat.pao2,
        pat.spo2,
        pat.fio2,
        pat.hr,
        pat.temp,
        pat.nbps,
        pat.nbpd,
        pat.nbpm,
        pat.abps,
        pat.abpd,
        pat.abpm,
        pat.rr,
        pat.tv,
        pat.pip,
        pat.plap,
        pat.mv
        --,row_number() OVER (PARTITION BY pat.icustay_id,pat.charttime ORDER BY s2.charttime DESC ) AS lastRowMAP
        ,
        CASE WHEN pat.charttime > s2.charttime
          THEN pat.charttime - s2.charttime
        ELSE s2.charttime - pat.charttime END AS sc,
        s2.map
      FROM stg14_1 pat
        LEFT JOIN map s2
          ON s2.icustay_id = pat.icustay_id
             AND (s2.charttime BETWEEN pat.charttime - INTERVAL '4' HOUR AND pat.charttime
                  OR s2.charttime BETWEEN pat.charttime AND pat.charttime + INTERVAL '4' HOUR)
      WHERE pat.lastRowMV = 1
  ), stg15_1 AS (
      SELECT
        *,
        row_number()
        OVER (
          PARTITION BY icustay_id, charttime
          ORDER BY sc ) AS lastRowMAP
      FROM stg15
  ), stg16 AS (
      SELECT
        pat.subject_id,
        pat.hadm_id,
        pat.icustay_id,
        pat.charttime,
        pat.pao2,
        pat.spo2,
        pat.fio2,
        pat.hr,
        pat.temp,
        pat.nbps,
        pat.nbpd,
        pat.nbpm,
        pat.abps,
        pat.abpd,
        pat.abpm,
        pat.rr,
        pat.tv,
        pat.pip,
        pat.plap,
        pat.mv,
        pat.map,
        row_number()
        OVER (
          PARTITION BY pat.icustay_id, pat.charttime
          ORDER BY s2.charttime DESC ) AS lastRowPEEP,
        s2.peep
      FROM stg15_1 pat
        LEFT JOIN peep s2
          ON s2.icustay_id = pat.icustay_id
             AND s2.charttime BETWEEN pat.charttime - INTERVAL '6' HOUR AND pat.charttime
      WHERE pat.lastRowMAP = 1
  )
  SELECT
    pat.subject_id,
    pat.hadm_id,
    pat.icustay_id,
    pat.charttime,
    pat.pao2,
    pat.spo2,
    pat.fio2,
    pat.hr,
    pat.temp,
    pat.nbps,
    pat.nbpd,
    pat.nbpm,
    pat.abps,
    pat.abpd,
    pat.abpm,
    pat.rr,
    pat.tv,
    pat.pip,
    pat.plap,
    pat.mv,
    pat.map,
    pat.peep
  FROM stg16 pat
  WHERE pat.lastRowPEEP = 1