# Building word cloud of most frequent words

wordcloud(words = wf$word, freq = wf$freq, min.freq = 4, max.words = 200, 
          random.order = F, rot.per = 0.40, 
          colors = brewer.pal(8, 'Accent'))

# build wordcloud using wordcloud2 library
wordcloud2(data = wf, figPath = figPath)
