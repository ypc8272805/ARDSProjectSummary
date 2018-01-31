CREATE TABLE patientschart_new AS WITH allpatients AS (
         SELECT select_patient.subject_id,
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
           FROM (mimiciii.select_patient_new
             LEFT JOIN mimiciii.pf_new ON ((select_patient.icustay_id = pf.icustay_id)))
        ), my_patients AS (
         SELECT allpatients.subject_id,
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
          WHERE ((allpatients.exclusion_pf = 0) AND (allpatients.exclusion_vent = 0) AND (allpatients.exclusion_los = 0) AND (allpatients.exclusion_first_stay = 0) AND (allpatients.exclusion_chest = 0) AND (allpatients.exclusion_age = 0))
        ), chart_value AS (
         SELECT ie.subject_id,
            ie.hadm_id,
            ie.icustay_id,
                CASE
                    WHEN (le.itemid = ANY (ARRAY[220277, 646])) THEN 'SPO2'::text
                    WHEN (le.itemid = ANY (ARRAY[220224, 779])) THEN 'PAO2'::text
                    WHEN (le.itemid = ANY (ARRAY[190, 3420, 223835, 3422])) THEN 'FIO2'::text
                    WHEN (le.itemid = ANY (ARRAY[211, 220045])) THEN 'HR'::text
                    WHEN (le.itemid = ANY (ARRAY[223762, 676, 677, 223761, 678, 679])) THEN 'TEMP'::text
                    WHEN (le.itemid = ANY (ARRAY[442, 455, 220179])) THEN 'NBPS'::text
                    WHEN (le.itemid = ANY (ARRAY[8440, 8441, 220180])) THEN 'NBPD'::text
                    WHEN (le.itemid = ANY (ARRAY[456, 443, 220181])) THEN 'NBPM'::text
                    WHEN (le.itemid = ANY (ARRAY[51, 6701, 220050])) THEN 'ABPS'::text
                    WHEN (le.itemid = ANY (ARRAY[8368, 8555, 220051])) THEN 'ABPD'::text
                    WHEN (le.itemid = ANY (ARRAY[52, 220052, 6702, 225312])) THEN 'ABPM'::text
                    WHEN (le.itemid = ANY (ARRAY[615, 618, 220210, 224690])) THEN 'RR'::text
                    WHEN (le.itemid = ANY (ARRAY[681, 682, 2400, 2408, 2534, 2420, 224685])) THEN 'TV'::text
                    WHEN (le.itemid = ANY (ARRAY[507, 535, 224695])) THEN 'PIP'::text
                    WHEN (le.itemid = ANY (ARRAY[543, 224696])) THEN 'PLAP'::text
                    WHEN (le.itemid = ANY (ARRAY[445, 448, 450, 224687])) THEN 'MV'::text
                    WHEN (le.itemid = ANY (ARRAY[444, 3502, 3503, 224697])) THEN 'MAP'::text
                    WHEN (le.itemid = ANY (ARRAY[506, 220339])) THEN 'PEEP'::text
                    ELSE NULL::text
                END AS label,
            ie.intime,
            ie.outtime,
            le.charttime,
            le.itemid,
                CASE
                    WHEN ((le.itemid = ANY (ARRAY[220277, 646])) AND (le.valuenum > (0)::double precision) AND (le.valuenum < (100)::double precision)) THEN le.valuenum
                    WHEN ((le.itemid = ANY (ARRAY[220224, 779])) AND (le.valuenum > (0)::double precision) AND (le.valuenum < (800)::double precision)) THEN le.valuenum
                    WHEN ((le.itemid = ANY (ARRAY[211, 220045])) AND (le.valuenum > (0)::double precision) AND (le.valuenum < (300)::double precision)) THEN le.valuenum
                    WHEN (le.itemid = 223835) THEN
                    CASE
                        WHEN ((le.valuenum > (0)::double precision) AND (le.valuenum <= (1)::double precision)) THEN (le.valuenum * (100)::double precision)
                        WHEN ((le.valuenum > (1)::double precision) AND (le.valuenum < (21)::double precision)) THEN NULL::double precision
                        WHEN ((le.valuenum >= (21)::double precision) AND (le.valuenum <= (100)::double precision)) THEN le.valuenum
                        ELSE NULL::double precision
                    END
                    WHEN ((le.itemid = ANY (ARRAY[3420, 3422])) AND (le.valuenum > (0)::double precision) AND (le.valuenum < (100)::double precision)) THEN le.valuenum
                    WHEN ((le.itemid = 190) AND (le.valuenum > (0.2)::double precision) AND (le.valuenum <= (1)::double precision)) THEN (le.valuenum * (100)::double precision)
                    WHEN ((le.itemid = ANY (ARRAY[223762, 676, 677])) AND (le.valuenum > (10)::double precision) AND (le.valuenum < (50)::double precision)) THEN le.valuenum
                    WHEN ((le.itemid = ANY (ARRAY[223761, 678, 679])) AND (le.valuenum > (70)::double precision) AND (le.valuenum < (120)::double precision)) THEN (((le.valuenum - (32)::double precision) * (5)::double precision) / (9)::double precision)
                    WHEN ((le.itemid = ANY (ARRAY[51, 442, 455, 6701, 220179, 220050])) AND (le.valuenum > (0)::double precision) AND (le.valuenum < (400)::double precision)) THEN le.valuenum
                    WHEN ((le.itemid = ANY (ARRAY[8368, 8440, 8441, 8555, 220180, 220051])) AND (le.valuenum > (0)::double precision) AND (le.valuenum < (300)::double precision)) THEN le.valuenum
                    WHEN ((le.itemid = ANY (ARRAY[456, 52, 6702, 443, 220052, 220181, 225312])) AND (le.valuenum > (0)::double precision) AND (le.valuenum < (300)::double precision)) THEN le.valuenum
                    WHEN ((le.itemid = ANY (ARRAY[615, 618, 220210, 224690])) AND (le.valuenum >= (0)::double precision) AND (le.valuenum < (70)::double precision)) THEN le.valuenum
                    WHEN ((le.itemid = ANY (ARRAY[681, 682, 2400, 2408, 2534, 2420, 224685])) AND (le.valuenum > (100)::double precision)) THEN le.valuenum
                    WHEN ((le.itemid = ANY (ARRAY[507, 535, 224695])) AND (le.valuenum > (0)::double precision) AND (le.valuenum < (100)::double precision)) THEN le.valuenum
                    WHEN ((le.itemid = ANY (ARRAY[543, 224696])) AND (le.valuenum > (0)::double precision) AND (le.valuenum < (100)::double precision)) THEN le.valuenum
                    WHEN ((le.itemid = ANY (ARRAY[445, 448, 450, 224687])) AND (le.valuenum > (0)::double precision)) THEN le.valuenum
                    WHEN ((le.itemid = ANY (ARRAY[444, 3502, 3503, 224697])) AND (le.valuenum > (0)::double precision) AND (le.valuenum < (100)::double precision)) THEN le.valuenum
                    WHEN ((le.itemid = ANY (ARRAY[506, 220339])) AND (le.valuenum > (0)::double precision) AND (le.valuenum < (40)::double precision)) THEN le.valuenum
                    ELSE NULL::double precision
                END AS valuenum,
            le.valueuom
           FROM (my_patients ie
             LEFT JOIN mimiciii.chartevents le ON (((le.subject_id = ie.subject_id) AND (le.hadm_id = ie.hadm_id) AND (ie.icustay_id = le.icustay_id) AND ((le.charttime >= ie.intime) AND (le.charttime <= ie.outtime)) AND (le.itemid = ANY (ARRAY[220277, 646, 220224, 779, 190, 3420, 3422, 223835, 211, 220045, 223762, 676, 677, 223761, 678, 679, 442, 455, 220179, 8440, 8441, 220180, 456, 443, 220181, 51, 6701, 220050, 8368, 8555, 220051, 52, 220052, 6702, 225312, 615, 618, 220210, 224690, 681, 682, 2400, 2408, 2534, 2420, 224685, 507, 535, 224695, 543, 224696, 445, 448, 450, 224687, 444, 3502, 3503, 224697, 506, 220339])))))
        )
 SELECT pvt.subject_id,
    pvt.hadm_id,
    pvt.icustay_id,
    pvt.charttime,
    max(
        CASE
            WHEN (pvt.label = 'SPO2'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS spo2,
    max(
        CASE
            WHEN (pvt.label = 'PAO2'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS pao2,
    max(
        CASE
            WHEN (pvt.label = 'FIO2'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS fio2,
    max(
        CASE
            WHEN (pvt.label = 'HR'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS hr,
    max(
        CASE
            WHEN (pvt.label = 'TEMP'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS temp,
    max(
        CASE
            WHEN (pvt.label = 'NBPS'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nbps,
    max(
        CASE
            WHEN (pvt.label = 'NBPD'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nbpd,
    max(
        CASE
            WHEN (pvt.label = 'NBPM'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nbpm,
    max(
        CASE
            WHEN (pvt.label = 'ABPS'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS abps,
    max(
        CASE
            WHEN (pvt.label = 'ABPD'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS abpd,
    max(
        CASE
            WHEN (pvt.label = 'ABPM'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS abpm,
    max(
        CASE
            WHEN (pvt.label = 'RR'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS rr,
    max(
        CASE
            WHEN (pvt.label = 'TV'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS tv,
    max(
        CASE
            WHEN (pvt.label = 'PIP'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS pip,
    max(
        CASE
            WHEN (pvt.label = 'PLAP'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS plap,
    max(
        CASE
            WHEN (pvt.label = 'MV'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS mv,
    max(
        CASE
            WHEN (pvt.label = 'MAP'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS map,
    max(
        CASE
            WHEN (pvt.label = 'PEEP'::text) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS peep
   FROM chart_value pvt
  GROUP BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime
  ORDER BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime;
