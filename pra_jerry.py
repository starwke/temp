sql = "SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'	concert_singer"

lstrip = sql.strip().split("\t")
print(lstrip)