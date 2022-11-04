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

    query = """
        SELECT * from game limit 1,10;
    """
    df = pandas.read_sql_query(query, sql_engine)
    print(df.head())


if __name__ == "__main__":
    sys.exit(main())
