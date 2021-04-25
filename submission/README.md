# N-SVAMT
Our team is from around the world and we think proteins are cool 🙌 Machine learning has a tremendous potential to solve apparent and invisible problems which may have an impact on a global scale. There is so much to explore, and hackathons like these get our hearts pumping.

### Project Description
We are working on the `optimal-ph` challenge. The gist of our logic lies in term frequency - inverse document frequency. A word vector represents a document as a list of numbers, with one for each possible word of the corpus. Vectorizing a sequence is taking the text and creating one of these vectors, and the numbers of the vectors somehow represent the content of the text. TF-IDF enables us to gives us a way to associate each word in a document with a number that represents how relevant each word is in that sequence.

To achieve this we first created custom classes from the mean pH for growth column. This turned our regression problem into a classification challenge, giving us more flexibility in terms of reducing the losses. It was a tremendous learning experience, under the pressure of maintaining a low RMSE and high Spearman correlation
.
To predict using our model, press "Open Application" on the left. 

### Scoreboard
You can track the performance of our predictor in the [challenge scoreboard](https://biolib.com/biohackathon/optimal-ph-scoreboard/).
