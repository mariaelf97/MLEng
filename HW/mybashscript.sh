#!/usr/bin/env bash

sleep 30

mysql -u root -pPassword123 -h mariadb -e "CREATE DATABASE IF NOT EXISTS baseball"
echo "baseball db created"
mysql -u root -pPassword123 -h mariadb baseball < /App/baseball.sql
echo "loaded baseball db data"
mysql -u root -pPassword123 -h mariadb baseball<mybaseball.sql > sql_output.txt
echo "file created"
head sql_output.txt



