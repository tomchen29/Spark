from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions

def loadMovieNames():
    movieNames = {}
    with open("ml-100k/u.ITEM") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

# Create a SparkSession (the config bit is only for Windows!)
spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("PopularMovies").getOrCreate()

# Get the raw data
lines = spark.sparkContext.textFile("ml-100k/u.data")
# Convert it to a RDD of Row objects
movies = lines.map(lambda x: Row(movieID =int(x.split()[1])))
# Convert that to a DataFrame
movieDataset = spark.createDataFrame(movies)

# Some SQL-style magic to sort all movies by popularity in one line!
movieDataset = movieDataset.groupBy("movieID").count()
movieDataset = movieDataset.orderBy("count", ascending=False).cache()

movieDataset.show()
# Show the results at this point:
#|movieID|count|
#+-------+-----+
#|     50|  584|
#|    258|  509|
#|    100|  508|

# Grab the top 10
top10 = movieDataset.take(10)
print ""

# Print the results by name
nameDict = loadMovieNames()  # Load up our movie ID -> name dictionary
for result in top10:
    # Each row has movieID, count as above.
    movie_name = nameDict[result[0]]
    occurences = result[1]
    print("%s: %d" % (movie_name, occurences))

# Stop the session
spark.stop()
