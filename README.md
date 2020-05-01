# README

## Introduction

In recent times, the novel corona virus has spread throughout the globe and put the world under lock down. In response, medical professionals have begun looking for solutions to this worldwide pandemic through all potential means, new and old. However, progress has been slow due to the lack of knowledge about the virus. Currently, there is an abundance of information in the form of academic papers for all topics, including those related to COVID-19, but being able to filter through this information quickly has been difficult. A multitude of organizations have come together to release CORD-19, an easy to parse data set containing thousands of research papers that are potentially connected to COVID-19. \par

The goal of this project was to explore the contents of the data set and establish similarities and connections between academic papers, as well as what kinds of topics are present in CORD-19. Various topic modeling techniques were used to establish the number of topics within the data set, as well as how coherent these topics were. The results of the project was an increase over the benchmark topic coherence score to .663, with a relative standard deviation of the topic coherence of 0.144.

## Data

The machine learning field that the solution is being designed for is under that of natural language processing and natural language understanding. For this project, the CORD-19 (version 8) was chosen. The data set was created by the US Government and large technology companies, and distributed through Kaggle in an easy to parse format.

## Implementation

For the models themselves, the hyper parameters passed through did not change much from the early models. The most important parameter for topic modeling is the number of topics, and like many other unsupervised clustering algorithms, the best way to determine the correct amount of clusters is to generate many models with different cluster amounts and compare them.



