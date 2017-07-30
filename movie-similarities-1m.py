import sys
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from math import sqrt

##the default executor memory budget is 512MB, not quiet sufficient for self-joining 1M movie ratings.
##so we set the memory to 1g isntead as below. 260 represents StarWar's ID:
#spark-submit --executor-memory 1g MovieSimilarities1M.py 260

def loadMovieNames():
    movieNames = {}
    with open("source/movies.dat") as f:
        for line in f:
            fields = line.split("::")
            movieNames[int(fields[0])] = fields[1].decode('ascii', 'ignore')
    return movieNames

def makePairs( userRatings ):
    #userRatings: [user_ID, ((movieID1, rating1), (movieID2, rating2))]
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return ((movie1, movie2), (rating1, rating2))

def filterDuplicates( userRatings ):
    #userRatings: [user_ID, ((movieID1, rating1), (movieID2, rating2))]
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    #return true if not equal
    return movie1 < movie2

def computeCosineSimilarity(ratingPairs):
    #ratingPairs: (rating1, rating2), (rating1, rating2), (rating1, rating2) ...
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1
    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)
    score = (numerator / (float(denominator))) if (denominator) else 0
    return (score, numPairs)

#use Spark built-in cluster manager to treat very laptop's core as a node
print("\nLoading movie names...")

#build a SparkContext nd create ratings: [user_ID, (movieID, rating)]
data = SparkContext(conf = SparkConf()).textFile("source/ratings.dat")
ratings = data.map(lambda l: l.split()).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))

# Emit every movie rated together by the same user.
# Self-join to find every combination.
joinedRatings = ratings.join(ratings)  #[_user_ID, ((movieID1, rating1), (movieID2, rating2))]

# Filter out duplicate pairs. filterDUplicates is a function that returns True of False
uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)

# Now key by movie pairs: [(movie1, movie2), (rating1, rating2)]
moviePairs = uniqueJoinedRatings.map(makePairs)

# We now have (movie1, movie2) => (rating1, rating2)
# Now collect all ratings for each movie pair and compute similarity
moviePairRatings = moviePairs.groupByKey()  #[(movie1, movie2), ((rating1, rating2), (rating1, rating2) ...)]

# Compute similarities.
moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity).cache() #[(movie1, movie2), (score, num_pairs)]

# Save the results if desired
#moviePairSimilarities.sortByKey()
#moviePairSimilarities.saveAsTextFile("movie-sims")

# Extract similarities for the movie we care about that are "good".
if __name__ == "__main__":
    movieID = int(sys.argv[1])
    scoreThreshold = 0.97        #cos similarity between the two relevant movies
    coOccurenceThreshold = 50    #at least 50 people have seen both movies

    # Select the target movie and filter for relevant movies that meet the threshold
    filteredResults = moviePairSimilarities.filter(lambda pairSim: \
        (pairSim[0][0] == movieID or pairSim[0][1] == movieID) \
        and pairSim[1][0] > scoreThreshold and pairSim[1][1] > coOccurenceThreshold)

    # Build a SparkSession and convert ratings to Dataframe
    spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName(
        "MovieSimilarities").getOrCreate()
    ratingsDataset = spark.createDataFrame(filteredResults)

    # Sort by quality score.
    results = filteredResults.map(lambda pairSim: (pairSim[1], pairSim[0])) #[(score, num_pairs), (movie1, movie2)]
    results = results.sortByKey(ascending = False).take(10)

    #load the full movie name by movie id
    nameDict = loadMovieNames()

    print("Top 10 similar movies for " + nameDict[movieID])
    movieset = set()
    for result in results:
        (sim, pair) = result
        # Display the similarity result that isn't the movie we're looking at
        similarMovieID = pair[0]
        if (similarMovieID == movieID):
            similarMovieID = pair[1]
        if similarMovieID not in movieset:
            print(str(similarMovieID)+"  "+nameDict[similarMovieID] + "\tscore: " + str(sim[0]) + "\t co-watch occurences: " + str(sim[1]))
            movieset.add(similarMovieID)

