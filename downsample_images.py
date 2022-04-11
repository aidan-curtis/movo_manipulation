f2 = open("slam_images.txt", "r")



i = 0
for line in f2:
    if i < 500:
        with open('slam_downsampled.txt', 'a') as f:
            f.write(line)
    else:
        if i % 5 == 0:
            with open('slam_downsampled.txt', 'a') as f:
                f.write(line)
    i+=1