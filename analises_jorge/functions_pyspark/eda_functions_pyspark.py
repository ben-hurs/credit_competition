import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import pyspark.sql.types as ST
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

import pyspark
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext


@F.udf(ST.StringType())
def perc_string(col):
    return str(round(col * 100, 2)) + "%"
  
@F.udf(ST.StringType())
def get_number_faixa(col):
    if col is not None:
        array_values = col.split("-")
        number = None
        if(array_values[0] == ""):
            number = "-"+array_values[1]
        else:
            number = array_values[0]
        return number
    else:
        return None
      
def get_numeric_cols(df):
  numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, ST.IntegerType) or isinstance(f.dataType, ST.FloatType) or isinstance(f.dataType, ST.DoubleType) or isinstance(f.dataType, ST.DecimalType) or isinstance(f.dataType, ST.LongType)]
  return numeric_cols

def calcular_ks_maximo(col_alvo, col_output_score, data_frame):

  import pyspark.sql.functions as F
  from pyspark.sql.window import Window
  import pyspark.sql.types as ST

  complete_window = Window.orderBy([col_output_score]).rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
  ord_window3 = Window.orderBy(col_output_score).rowsBetween(Window.unboundedPreceding, Window.currentRow)

  ks_step1 = data_frame.where(F.col(col_output_score).isNotNull()).withColumn(col_output_score, F.col(col_output_score).cast(ST.DoubleType()))
  ks_step2 = ks_step1.groupBy(col_output_score).agg(F.sum(col_alvo).alias('QTD_MAU'),F.count(col_alvo).alias('QTD_TOTAL'))
  ks_step3 = ks_step2.select(F.col('*'),(F.col('QTD_TOTAL') - F.col('QTD_MAU')).alias('QTD_BOM'))
  ks_step4 = ks_step3.select(F.col('*'),(F.sum('QTD_MAU').over(ord_window3)).alias('MAU_ACUMULADO'),        (F.sum('QTD_BOM').over(ord_window3)).alias('BOM_ACUMULADO'))
  
  ks_step5 = ks_step4.select(F.col('*'),(F.col('QTD_MAU') / F.max('MAU_ACUMULADO').over(complete_window)).alias('TX_MAU_TOTAL'),(F.col('QTD_BOM') / F.max('BOM_ACUMULADO').over(complete_window)).alias('TX_BOM_TOTAL'))
  
  ks_step6 = ks_step5.select(F.col('*'),(F.sum('TX_MAU_TOTAL').over(ord_window3)).alias('TX_MAU_ACUM'), (F.sum('TX_BOM_TOTAL').over(ord_window3)).alias('TX_BOM_ACUM'))
  
  ks_step7 = ks_step6.withColumn("KS", F.abs(F.col("TX_BOM_ACUM")-F.col("TX_MAU_ACUM"))*100.0)
  
  

  ks_max = ks_step7.agg({'KS' : 'max'}).collect()[0][0]
  return ks_max

import numpy as np

def intervals_quantile_column(df, col_value, quantile, alias_column="FAIXA"):

  df = df.withColumn(col_value, F.col(col_value).cast(ST.FloatType()))

  perc = 1.0 / quantile
  lista_percentis = list(np.arange(0.0, 1.0, perc))
  lista_percentis.append(1.0)
  x = df

  lista_valores = x.approxQuantile(col_value, lista_percentis, 0)
  data = lista_valores   
  data = data[::-1]
  when_builder = None
  for j in range(len(data) -1):
    atual = data[j+1]
    post = data[j]
    value = str(round(atual,3)) + "-" + str(round(post,3))

    if j < (len(data) - 1):
      if j == 0:
        when_builder = F.when((F.col(col_value) >= atual) & (F.col(col_value) <=post), value).otherwise(None)
      else:
        when_builder = F.when((F.col(col_value) >= atual) & (F.col(col_value) < post), value).otherwise(when_builder)

  df = df.withColumn(alias_column,when_builder)
  return df

def set_intervall(df,column,interval_list, alias_column="FAIXA"):

  df = df.withColumn(column, F.col(column).cast(ST.FloatType()))

  data = interval_list   
  data = data[::-1]
  when_builder = None
  for j in range(len(data) -1):
    atual = data[j+1]
    post = data[j]
    value = str(round(atual,3)) + "-" + str(round(post,3))
    
    if j < (len(data) - 1):
      if j == 0:
        when_builder = SF.when((SF.col(column) >= atual) & (SF.col(column) <=post), value).otherwise(None)
      else:
        when_builder = SF.when((SF.col(column) >= atual) & (SF.col(column) < post), value).otherwise(when_builder)

  df = df.withColumn(alias_column,when_builder)#.otherwise(c)))#.otherwise(SF.when(j != (len(data) - 2), b).otherwise(c)))
  return df



def eda_numeric_columns(spark, df, target_col=None):

  #describe padrão
  numeric_cols = get_numeric_cols(df)
  
  print("Total de {} colunas numéricas".format(str(len(numeric_cols))))
  
  if (target_col) and (isinstance(target_col, str)):
      numeric_cols.append(target_col)
  
  df_numerics = df.select(numeric_cols)
  
  df_eda_numerics = df_numerics.describe()
  
  #Percentual de nulos
  count_df = df.count()
  df_perc_null = (df_numerics.select([perc_string((F.count(F.when(F.isnull(c), c)) / count_df)).alias(c) for c in df_numerics.columns])
                 .withColumn("summary",F.lit("Perc Null")))
  
  df_eda_numerics = df_eda_numerics.unionByName(df_perc_null)
  
  #Percentis das variáveis numéricas
  schema_percentile = ST.StructType([ \
    ST.StructField("p10", ST.DoubleType(),True), \
    ST.StructField("p20", ST.DoubleType(),True), \
    ST.StructField("p30", ST.DoubleType(),True), \
    ST.StructField("p40", ST.DoubleType(), True), \
    ST.StructField("p50", ST.DoubleType(), True), \
    ST.StructField("p60", ST.DoubleType(),True), \
    ST.StructField("p70", ST.DoubleType(),True), \
    ST.StructField("p80", ST.DoubleType(),True), \
    ST.StructField("p90", ST.DoubleType(), True), \
    ST.StructField("p95", ST.DoubleType(), True), \
    ST.StructField("col", ST.StringType(), True), \
  ])
  
  schema_ks = ST.StructType([ \
    ST.StructField("summary", ST.StringType(), True), \
  ])
  df_percentile = spark.createDataFrame([],schema=schema_percentile)
#   df_ks = spark.createDataFrame([])
  
#   df_ks.display()
  dict_df_ks ={"summary":"ks"}
  for col in numeric_cols:
    #Percentis
    percentile_10 = F.expr('percentile_approx('+ col +', 0.1)')
    percentile_20 = F.expr('percentile_approx('+ col +', 0.2)')
    percentile_30 = F.expr('percentile_approx('+ col +', 0.3)')
    percentile_40 = F.expr('percentile_approx('+ col +', 0.4)')
    percentile_50 = F.expr('percentile_approx('+ col +', 0.5)')
    percentile_60 = F.expr('percentile_approx('+ col +', 0.6)')
    percentile_70 = F.expr('percentile_approx('+ col +', 0.7)')
    percentile_80 = F.expr('percentile_approx('+ col +', 0.8)')
    percentile_90 = F.expr('percentile_approx('+ col +', 0.9)')
    percentile_95 = F.expr('percentile_approx('+ col +', 0.95)')
    
    df_percentile_temp = df_numerics.agg(percentile_10.alias('p10'),
                          percentile_20.alias('p20'),
                          percentile_30.alias('p30'),
                          percentile_40.alias('p40'),
                          percentile_50.alias('p50'),
                          percentile_60.alias('p60'),
                          percentile_70.alias('p70'),
                          percentile_80.alias('p80'),
                          percentile_90.alias('p90'),
                          percentile_95.alias('p95')
                          ).withColumn("col",F.lit(col))
    
    df_percentile = df_percentile.unionByName(df_percentile_temp)
    
    #ks
    if (target_col):
      ks_var = calcular_ks_maximo(target_col,col,df_numerics)
#     df_ks = df_ks.withColumn(col,F.lit(ks_var))
      dict_df_ks[col] = ks_var
  
  
  df_percentile_final_pandas = df_percentile.toPandas().set_index("col").transpose()
  df_percentile_final_pandas["summary"] = ["p10","p20","p30","p40","p50","p60","p70","p80","p90","p95"]
  df_percentile_final = spark.createDataFrame(df_percentile_final_pandas)
  
  df_eda_numerics = df_eda_numerics.unionByName(df_percentile_final)
  if (target_col):
    df_ks = spark.createDataFrame([dict_df_ks])
    df_eda_numerics = df_eda_numerics.unionByName(df_ks)
  
  
  return df_eda_numerics
    
  
#   quantile_50 = F.expr('percentile_approx('+ 'resulting_balance' +', 0.5)')
    
  
def eda_categoric_columns(spark, df, col_alvo=None):

  categoric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, ST.StringType)]   

  print("Total de {} colunas categoricas".format(str(len(categoric_cols))))

  if (col_alvo is not None):
    categoric_cols.append(col_alvo)
    df = df.select(categoric_cols)
  else:
    df = df.select(categoric_cols)


  df_cat = None

  if (col_alvo is not None):
    for x in categoric_cols:
      if df_cat is None:
        df_cat = df.select(
            F.lit(x).alias('variable_name'),
            F.col(x).cast('string').alias('variable_value'),
            F.col(col_alvo).alias('target')
        )
        
        continue
        

      else:
        df_cat = df_cat.unionByName(
            df.select(
                F.lit(x).alias('variable_name'),
                F.col(x).cast('string').alias('variable_value'),
                F.col(col_alvo).alias('target')
            )
        )
        
        continue

    if df_cat is not None:
      df_cat = (df_cat.groupBy('variable_name', 'variable_value')
                .agg(F.count(F.when(F.isnull("variable_value") | ~F.isnull("variable_value"), "variable_value")).alias('QTD_TOTAL_VARIABLE'),
                     F.count(F.when(F.isnull("variable_value"), "variable_value")).alias('QTD_TOTAL_VARIABLE_NULL'),
                     F.count(F.when(F.isnull("target") | ~F.isnull("target"), "target")).alias('QTD_TARGET_TOTAL'),
                     F.count(F.when(F.isnull("target"), "target")).alias("QTD_TARGET_NULL"),
                     F.sum('target').alias('QTD_TARGET_MAU'),
      ))

      df_cat = df_cat.withColumn("PERC_NULL", perc_string((F.col("QTD_TOTAL_VARIABLE_NULL") / F.col("QTD_TOTAL_VARIABLE"))))
      df_cat = df_cat.withColumn("QTD_TARGET_BOM",F.col('QTD_TARGET_TOTAL') - F.col('QTD_TARGET_MAU') - F.col('QTD_TARGET_NULL'))
      df_cat = df_cat.withColumn("PERC_TARGET_NULL", perc_string((F.col("QTD_TARGET_NULL") / F.col("QTD_TARGET_TOTAL"))))
      df_cat = df_cat.withColumn("PERC_TARGET_BOM", perc_string((F.col("QTD_TARGET_BOM") / F.col("QTD_TARGET_TOTAL"))))
      df_cat = df_cat.withColumn("PERC_TARGET_MAU", perc_string((F.col("QTD_TARGET_MAU") / F.col("QTD_TARGET_TOTAL"))))
        
  else:
    for x in categoric_cols:
      if df_cat is None:
        df_cat = df.select(
          F.lit(x).alias('variable_name'),
          F.col(x).cast('string').alias('variable_value'),
        )
      else:
        df_cat = df_cat.unionByName(
          df.select(
          F.lit(x).alias('variable_name'),
          F.col(x).cast('string').alias('variable_value'),
          )
        )
    if df_cat is not None:
      df_cat = (df_cat.groupBy('variable_name', 'variable_value')
      .agg(
      F.count(F.when(F.isnull("variable_value") | ~F.isnull("variable_value"), "variable_value")).alias('QTD_TOTAL'),

      ))

  return df_cat.orderBy("variable_name", "variable_value")

def binary_columns_cross_target(df, target_col, col):
  df_analise = df.groupBy(col).pivot(target_col).count()
  
  count_df = df.count()
  df_analise = (df_analise
               .withColumn("Total",F.col("1")+F.col("0"))
               .withColumn("Percentual Total",(F.col("Total")/F.lit(count_df))*100)
               .withColumn("Percentual Mau",((F.col("1"))/(F.col("Total")))*100)
               .withColumn("Percentual Bom",((F.col("0"))/(F.col("Total")))*100)
               )
  return df_analise

def continuous_columns_cross_target(df, target_col, col,quantile):
  alias_columns = "FAIXA_{}".format(col)
  df_quantile = intervals_quantile_column(df,col,quantile,alias_column=alias_columns)
  df_quantile = df_quantile.groupBy(alias_columns).pivot("is_fraudster").count()
  df_quantile = df_quantile.withColumn("ORDER_FAIXA",get_number_faixa(alias_columns)).withColumn("ORDER_FAIXA", F.col("ORDER_FAIXA").cast(ST.DoubleType()))
  df_quantile = df_quantile.orderBy(F.col("ORDER_FAIXA").asc()).drop("ORDER_FAIXA")
  
  count_df = df.count()
  df_quantile = (df_quantile
               .withColumn("Total",F.col("1")+F.col("0"))
               .withColumn("Percentual Total",(F.col("Total")/F.lit(count_df))*100)
               .withColumn("Percentual Mau",((F.col("1"))/(F.col("Total")))*100)
               .withColumn("Percentual Bom",((F.col("0"))/(F.col("Total")))*100)
               )
  return df_quantile

