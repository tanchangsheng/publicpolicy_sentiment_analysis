'''
Consolidate data from google docs. Combine all threads and comments to single post.
'''
import sys
import unicodedata


def combine(data, origin_pos, text_pos, topic_pos, topics_str, output):


    fin = open(data, 'r')
    fout = open(output, 'w')

    next(fin)
    pid = None
    for line in fin:
        # split line into columns
        line = line.split('\t')

        # obtain the source of post/comment
        origin = line[int(origin_pos)]

        # obtain the text body of the post/content
        text = line[int(text_pos)]

        # obain the topic(s) assigned to post/comment as list
        this_topics = line[int(topic_pos)].lower().split(",")

        # topics of interest to findings as a list
        topics = topics_str.split(',')

        # remove unicode words (e.g.)
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')

        # converts stream of bytes to utf-8 encoded string
        text = text.decode("utf-8")

        # if any of labelled topics are in interested topics
        add_row = False
        for topic in this_topics:
            if topic in topics:
                add_row = True
                break
        # write to output file
        if add_row:
            fout.write(origin + '\t' + str(this_topics) + '\t' + text + '\n')

    fin.close()
    fout.close()

if __name__ == '__main__':

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        print ("Please provide all 6 arguments")
        sys.exit()

    data = sys.argv[1]
    origin_pos = sys.argv[2]
    text_pos = sys.argv[3]
    topic_pos = sys.argv[4]
    topics = sys.argv[5]
    output = sys.argv[6]

    combine(data, origin_pos, text_pos, topic_pos, topics, output)
