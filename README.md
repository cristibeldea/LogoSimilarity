# LogoSimilarity
This is a program that takes a list of domains and downloads their logos, classifying them by visual similarity, without using any sort of machine learning model.

# Description
This challenge implied taking a list of 3000+ domains and building a program that searches for the logos of those domains, successfully downloading them for at least 97% of the lines in the list. After that, the logos should be clustered by similarity, without using any sort of machine learning model.

# Content
The project contains the following files:
  -logo_extraction.py which takes care of reading the domain list and downloading the logos
  -logo_clustering.py which clusters the images by similarity in separate folders
  -logos.snappy.parquet - the domain list

When the user runs logo_extraction.py, a folder called "logos" will be created, where the photos of the logos will be downloaded
When the user runs logo_clustering.py, 2 other folders will be created ("grouped_logos" and "processed_bin") containing the resulted
clusters and the photos in binary form representing the main shapes, respectively.

# Main functionality
(logo_extraction.py)
To ensure a success rate of 97.22% in extracting and downloading the logos, the logo extraction relies on 3 "safety nets":
  -it firstly accesses the main page of the website, searching in the HTML document for words like "logo"
  -if that doesnt work, it uses an online service called Clearbit which provides us with a public API containing informations about the companies, including the logo
  -finally, if the first two methods fail, the program falls back to extracting the favicon as a .ico file (the small pictogram near the name of the tab)
The program reads the "domain" column of the .parquet file, eliminating the duplicates and null values. Afterwards, it uses ThreadPoolExecutor for running the logo extraction methods in parallel over 20 threads,
ensuring fast download time. In the end the logos are saved as mostly .png files in a folder called "logos" and it prints the download success rate in Terminal.

(logo_clustering.py)
The program uses a set of 3 filters to find similarities between the photos present in the "logos" folder:
  -a hash based filter: which creates a "visual footprint" of the image, based on the brightness difference between neighbor pixels. It returns a 64 bit hash.
  -a shape based filter: which takes each photo and normalizes them by area (scaling down the bigger logo to the size of the smaller one, reducing pixeled margins), transforms it into a binary image (black and white), inverting the images so the shape remains black and full, and the background white. It aligns both shapes on a white canvas and compares the shapes with XOR, highlighting only the different pixels. In the end it computes a similarity score where 0 si identical and 1 is completely different.
  -a mean color based filter: starts by converting the images to HSV, creates a mask to ignore white (or clode to white) pixels (which is usually the background and we dont want the mean color to be "diluted"). Then, it computes the mean color for each image and calculates the Euclidean distance between the 2 color vectors. The result is a normalized number (in the [0, 1] interval) representing the similarity score.
  
The program starts by calculating the hash filter. If the distance resulted from this filter is too big, the other two filters arent perdormed to save computing time. If the logos pass the hash filter, the next two filters are calculated and the resulting distance is a weighted sum between the two distances of the two filters. The weights are chosen experimentally.
In the end, we create a distance matrix, and based on that we group the logos if the distance between them is less than a threshold that is also chosen experimentally. If a logo is already grouped in a cluster, we dont count it again to find similar logos to it, because that leads to chain linking the logos (if A is similar to B and B is similar to C, that would result in A is similar to C, which is not always true).
The photos are saved in folders according to their clusters.

Thank you for your time and consideration!
Dan Cristian Beldea


