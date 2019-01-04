import imageio
images = []

seriesLength = 9

folder = "../plots/series/"
filenames = [folder+""+str(x)+".jpg" for x in range(0,seriesLength)]

for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave(folder+'metropolis_random.gif', images, duration=0.25)