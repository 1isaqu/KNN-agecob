-- ====================================================================
-- gerar_dataset_v2.sql — d.csv Phase 2 retreino
-- 10 cols, exportar SSMS Save Results As CSV (sep ;)
-- Janela 2025-11-07 -> 2026-05-08, dias úteis, horas 8-19
-- AUTOS + CONSUMER, filtro agentes completo, zeros incluídos
-- ====================================================================

SET DATEFIRST 7;
SET NOCOUNT ON;

DECLARE @data_ini DATE = '2025-11-07';
DECLARE @data_fim DATE = '2026-05-08';

-- Tally table dias úteis (sem CTE recursivo)
IF OBJECT_ID('tempdb..#dias') IS NOT NULL DROP TABLE #dias;
CREATE TABLE #dias (dia DATE PRIMARY KEY, dia_semana INT);

DECLARE @d DATE = @data_ini;
WHILE @d <= @data_fim
BEGIN
    IF DATEPART(WEEKDAY, @d) BETWEEN 2 AND 6
        INSERT INTO #dias(dia, dia_semana) VALUES (@d, DATEPART(WEEKDAY, @d));
    SET @d = DATEADD(DAY, 1, @d);
END;

-- Horas
IF OBJECT_ID('tempdb..#horas') IS NOT NULL DROP TABLE #horas;
CREATE TABLE #horas (hora INT PRIMARY KEY);
INSERT INTO #horas VALUES (8),(9),(10),(11),(12),(13),(14),(15),(16),(17),(18),(19);

-- Grade: banco x dia x hora
IF OBJECT_ID('tempdb..#grade') IS NOT NULL DROP TABLE #grade;
CREATE TABLE #grade (banco_origem VARCHAR(30), dia DATE, dia_semana INT, hora INT,
                     PRIMARY KEY (banco_origem, dia, hora));

INSERT INTO #grade
SELECT b.banco_origem, d.dia, d.dia_semana, h.hora
FROM (SELECT 'COBwebRCBAUTOS' AS banco_origem UNION ALL SELECT 'COBwebRCBCONSUMER') b
CROSS JOIN #dias d
CROSS JOIN #horas h;

-- Acordos CONSUMER
IF OBJECT_ID('tempdb..#acordos') IS NOT NULL DROP TABLE #acordos;
CREATE TABLE #acordos (banco_origem VARCHAR(30), dia DATE, hora INT, qtd INT,
                       PRIMARY KEY (banco_origem, dia, hora));

INSERT INTO #acordos
SELECT
    'COBwebRCBCONSUMER',
    CAST(R.DT_EMISSAO AS DATE),
    DATEPART(HOUR, R.DT_EMISSAO),
    COUNT(DISTINCT R.NR_RECEBIMENTO)
FROM COBwebRCBCONSUMER..REC_MASTER R WITH (NOLOCK)
INNER JOIN COBwebRCBCONSUMER..USU_MASTER U WITH (NOLOCK)
    ON U.ID_USUARIO = R.ID_USUARIO
WHERE R.DT_EMISSAO >= @data_ini
  AND R.DT_EMISSAO <  DATEADD(DAY, 1, @data_fim)
  AND R.ID_REC_STATUS IN (1, 3, 12)
  AND R.PARCELA = 0
  AND DATEPART(HOUR, R.DT_EMISSAO) BETWEEN 8 AND 19
  AND UPPER(LTRIM(RTRIM(U.NOME)))  <> 'COBDESANTOS'
  AND UPPER(LTRIM(RTRIM(U.NOME)))  <> 'NEMBUSUSER'
  AND UPPER(LTRIM(RTRIM(U.CHAVE))) <> 'NEMBUSUSER'
  AND UPPER(LTRIM(RTRIM(U.NOME)))  NOT LIKE 'ANTLIA%'
  AND UPPER(LTRIM(RTRIM(U.NOME)))  NOT LIKE 'INTERNA%'
  AND UPPER(LTRIM(RTRIM(U.CHAVE))) NOT LIKE 'INTERNA%'
  AND UPPER(LTRIM(RTRIM(U.CHAVE))) NOT LIKE 'SUPORTE%'
  AND UPPER(LTRIM(RTRIM(U.CHAVE))) NOT LIKE 'SISTEMA%'
GROUP BY CAST(R.DT_EMISSAO AS DATE), DATEPART(HOUR, R.DT_EMISSAO);

INSERT INTO #acordos
SELECT
    'COBwebRCBAUTOS',
    CAST(R.DT_EMISSAO AS DATE),
    DATEPART(HOUR, R.DT_EMISSAO),
    COUNT(DISTINCT R.NR_RECEBIMENTO)
FROM COBwebRCBAUTOS..REC_MASTER R WITH (NOLOCK)
INNER JOIN COBwebRCBAUTOS..USU_MASTER U WITH (NOLOCK)
    ON U.ID_USUARIO = R.ID_USUARIO
WHERE R.DT_EMISSAO >= @data_ini
  AND R.DT_EMISSAO <  DATEADD(DAY, 1, @data_fim)
  AND R.ID_REC_STATUS IN (1, 3, 12)
  AND R.PARCELA = 0
  AND DATEPART(HOUR, R.DT_EMISSAO) BETWEEN 8 AND 19
  AND UPPER(LTRIM(RTRIM(U.NOME)))  <> 'COBDESANTOS'
  AND UPPER(LTRIM(RTRIM(U.NOME)))  <> 'NEMBUSUSER'
  AND UPPER(LTRIM(RTRIM(U.CHAVE))) <> 'NEMBUSUSER'
  AND UPPER(LTRIM(RTRIM(U.NOME)))  NOT LIKE 'ANTLIA%'
  AND UPPER(LTRIM(RTRIM(U.NOME)))  NOT LIKE 'INTERNA%'
  AND UPPER(LTRIM(RTRIM(U.CHAVE))) NOT LIKE 'INTERNA%'
  AND UPPER(LTRIM(RTRIM(U.CHAVE))) NOT LIKE 'SUPORTE%'
  AND UPPER(LTRIM(RTRIM(U.CHAVE))) NOT LIKE 'SISTEMA%'
GROUP BY CAST(R.DT_EMISSAO AS DATE), DATEPART(HOUR, R.DT_EMISSAO);

-- Batimentos por banco
IF OBJECT_ID('tempdb..#batim') IS NOT NULL DROP TABLE #batim;
CREATE TABLE #batim (banco_origem VARCHAR(30), dia_batimento DATE,
                     PRIMARY KEY (banco_origem, dia_batimento));

INSERT INTO #batim
SELECT DISTINCT 'COBwebRCBCONSUMER', CAST([DATA] AS DATE)
FROM COBwebRCBCONSUMER..CARGA_LOTE WITH (NOLOCK)
WHERE ID_USUARIO = 1 AND QTD_NV_CLI > 10000 AND [DATA] <= @data_fim;

INSERT INTO #batim
SELECT DISTINCT 'COBwebRCBAUTOS', CAST([DATA] AS DATE)
FROM COBwebRCBAUTOS..CARGA_LOTE WITH (NOLOCK)
WHERE ID_USUARIO = 1 AND QTD_NV_CLI > 10000 AND [DATA] <= @data_fim;

-- SELECT FINAL — 10 colunas
WITH base AS (
    SELECT
        g.banco_origem,
        g.dia,
        g.dia_semana,
        g.hora,
        COALESCE(a.qtd, 0) AS qtd
    FROM #grade g
    LEFT JOIN #acordos a
      ON a.banco_origem = g.banco_origem
     AND a.dia          = g.dia
     AND a.hora         = g.hora
),
agg AS (
    SELECT
        banco_origem, dia, dia_semana, hora, qtd,
        SUM(qtd) OVER (PARTITION BY banco_origem, dia) AS total_dia,
        SUM(qtd) OVER (PARTITION BY banco_origem, dia ORDER BY hora
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS acumulado_ate_hora
    FROM base
),
com_batim AS (
    SELECT
        a.*,
        (SELECT MAX(dia_batimento) FROM #batim b
         WHERE b.banco_origem = a.banco_origem AND b.dia_batimento <= a.dia) AS data_ultimo_batimento
    FROM agg a
)
SELECT
    banco_origem,
    CONVERT(VARCHAR(10), dia, 23) AS dia,
    hora,
    dia_semana,
    total_dia,
    acumulado_ate_hora,
    CASE WHEN total_dia = 0 THEN CAST(0 AS FLOAT)
         ELSE CAST(acumulado_ate_hora AS FLOAT) / total_dia
    END AS proporcao_ate_hora,
    CONVERT(VARCHAR(10), data_ultimo_batimento, 23) AS data_ultimo_batimento,
    DATEDIFF(DAY, data_ultimo_batimento, dia)       AS dias_desde_ultimo_batimento,
    CASE
        WHEN data_ultimo_batimento IS NULL THEN 'basal'
        WHEN DATEDIFF(DAY, data_ultimo_batimento, dia) BETWEEN 0 AND 5  THEN 'pos_batimento'
        WHEN DATEDIFF(DAY, data_ultimo_batimento, dia) BETWEEN 6 AND 15 THEN 'absorcao'
        ELSE 'basal'
    END AS faixa_batimento
FROM com_batim
ORDER BY banco_origem, dia, hora;

-- Cleanup
DROP TABLE #dias, #horas, #grade, #acordos, #batim;
