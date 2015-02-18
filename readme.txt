This is a project for coding narratives for themes using ML

I'm using narratives 1-140 for investigation, with 130-164 used for testing later

Also, it's worth noting that you can recieve a score of 1.5 for agency, but the data set has none of that...

Preprocessing:
remove any text in parentheses (because someone added unwanted coded in the narratives)
remove non-alphabet characters except for spaces
fold everything down to lower case

I'm also stemming words using the snowball (porter2) stemmer. This means that dog and dogs for example will be the same