import sys

import pandas
import sqlalchemy


def main():
    db_user = ""
    db_pass = ""  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )
    # pragma: allowlist secret

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """    select
                     bs.winner_home_or_away
                     ,bc.team_id
                     ,bc.game_id
                     ,bc.homeTeam
                     ,bc.awayTeam
                     ,bc.Hit
                     ,bc.atBat
                     ,bc.Hit_By_Pitch
                     ,bc.Home_run
                     ,bc.Strikeout
                     ,bc.Single
                     ,bc.Double
                     ,bc.Triple
                     ,bc.Sac_Fly
                     ,sum(bc.Home_run)/sum(nullif(bc.Hit,0)) as Home_run_per_hit
             from batter_counts bc, boxscore bs
             where bs.game_id=bc.game_id and bs.winner_home_or_away = "H"
             group by game_id, team_id;
    """
    df = pandas.read_sql_query(query, sql_engine)
    return df


if __name__ == "__main__":
    sys.exit(main())
