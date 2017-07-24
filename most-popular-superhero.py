from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("PopularHero")
sc = SparkContext(conf = conf)

def countCoOccurences(line):
    elements = line.split()
    return (int(elements[0]), len(elements) - 1)

def parseNames(line):
    fields = line.split('\"')
    return (int(fields[0]), fields[1].encode("utf8"))

names = sc.textFile("source\marvel-names.txt")
namesRdd = names.map(parseNames)

lines = sc.textFile("source\marvel-graph.txt")

#map input data to (heroID, num_of_co_occurences)
pairings = lines.map(countCoOccurences)

#add up co-occurence
totalFriendsByCharacter = pairings.reduceByKey(lambda x, y : x + y)

#flip
flipped = totalFriendsByCharacter.map(lambda c : (c[1], c[0]))

mostPopular = flipped.max()

#look up the corresponding superhero's name by id
mostPopularName = namesRdd.lookup(mostPopular[1])[0]

print(str(mostPopularName) + " is the most popular superhero, with " + \
    str(mostPopular[0]) + " co-appearances.")
