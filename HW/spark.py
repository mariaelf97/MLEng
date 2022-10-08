import sys

from pyspark import StorageLevel
from pyspark.ml.feature import SQLTransformer
from pyspark.sql import SparkSession


def main():
    appName = "baseball mariadb spark"
    master = "local"
    spark = SparkSession.builder.appName(appName).master(master).getOrCreate()

    user = "Spark"
    password = "Maryam"

    jdbc_url = "jdbc:mysql://localhost:3306/baseball?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    game = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "baseball.game")
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    game.createOrReplaceTempView("game")
    game.persist(StorageLevel.DISK_ONLY)

    batter_counts = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "baseball.batter_counts")
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    batter_counts.createOrReplaceTempView("batter_counts")
    batter_counts.persist(StorageLevel.DISK_ONLY)

    joined_table = spark.sql(
        """
    select
            g.local_date
            , bc.batter
            , bc.game_id
            , bc.Hit
            , bc.atBat
    from batter_counts bc
    join game g on bc.game_id = g.game_id
    ;
    """
    )
    joined_table.createOrReplaceTempView("joined_table")
    joined_table.persist(StorageLevel.DISK_ONLY)

    SQL_QUERY = """
        select    abg1.batter
                , abg1.local_date
                , abg1.Hit+sum(coalesce(abg2.Hit,0)) total_Hit
                , abg1.atBat+sum(coalesce(abg2.atBat,0)) total_atBat
        from joined_batter_game abg1 force index (batter_indx)
        left join joined_batter_game abg2 on abg1.batter=abg2.batter
        and timestampdiff(day,abg1.local_date,abg2.local_date)<=99 and timestampdiff(day,abg1.local_date,abg2.local_date)>0
        group by abg1.batter,abg1.local_date,abg1.Hit, abg1.atBat
        order by 1,2
    ;
    """

    sqlTrans = SQLTransformer().setStatement(SQL_QUERY)
    sqlTrans.transform(joined_table).show()


if __name__ == "__main__":
    sys.exit(main())
