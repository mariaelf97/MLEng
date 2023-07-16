#!/usr/bin/env bash

sleep 30

  mysql -u root -pPassword123 -h mariadb -e "CREATE DATABASE IF NOT EXISTS baseball"
echo "baseball db created, now loading the baseball db...it's very slow..slower than molasses"
mysql -u root -pPassword123 -h mariadb baseball < /app/baseball.sql
echo "loaded baseball db data, now running sql...please be patient..maybe get some tea.."
mysql -u root -pPassword123 -h mariadb baseball<batting_average.sql
echo "SQL table made successfully, now running the python code.... it's going to be very slow.."
python3 basbeall_fresh.py



