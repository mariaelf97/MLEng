
-- first join the game and batter counts table and save the results as a temporary table
drop temporary table if exists joined_batter_game;
create temporary table joined_batter_game
select game.local_date,batter_counts.batter,batter_counts.game_id,batter_counts.Hit,batter_counts.atBat from batter_counts join game on batter_counts.game_id=game.game_id;


-- change batter data type to character because it was integer
alter table joined_batter_game modify batter varchar(100);


-- calculate annual batting average
drop temporary table if exists annual_at_bat_per_player;
create temporary table annual_at_bat_per_player
select batter,year(local_date) as date_year, sum(Hit)/sum(nullif(atBat,0)) as Average_at_bat from joined_batter_game group by batter,year(local_date) order by year(local_date);
-- view the table annual_at_bat_per_player
select * from annual_at_bat_per_player order by batter limit 1,20;

-- calculate historic batting average
drop temporary table if exists historic_at_bat_per_player;
create temporary table historic_at_bat_per_player
select batter, sum(Hit)/sum(nullif(atBat,0)) as Average_at_bat from joined_batter_game group by batter;
-- view the table historic_at_bat_per_player
select * from historic_at_bat_per_player order by batter limit 1,20;


-- calculate last 100 days batting average --
-- first we create a table with the 99 recent entries for each player --
drop temporary table if exists last_100_days_per_player;
create temporary table last_100_days_per_player
select * from (select * , row_number() over (partition by batter order by local_date desc) as date_rank from joined_batter_game) ranks where date_rank <=99;
-- Then we calculate the running sum for last 99 days (not including the last day) for each player
drop temporary table if exists last_100_days_running_sum_per_player;
create temporary table last_100_days_running_sum_per_player
select batter,local_date,Hit,atBat,sum(atBat) over (partition by batter order by local_date) as sum_atBat,sum(Hit) over (partition by batter order by local_date) as sum_hit from last_100_days_per_player;
-- The last thing we need to do is to divide the hit sum by atBat sum
drop temporary table if exists last_100_days_running_sum_average_per_player;
create temporary table last_100_days_running_sum_average_per_player
select batter,local_date,sum_atBat,sum_hit, sum_hit/nullif(sum_atBat,0) as batting_average from last_100_days_running_sum_per_player;
-- show the results --
select * from last_100_days_running_sum_average_per_player limit 1,20;
