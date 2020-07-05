# A-dataset-of-Indian-Folk-Songs

This repository provides audio samples (5 sec
chunks) for Indian folk songs classification. Folk
songs from following five regions of India are used
in this research.

<ol>
  <li>Assamese from Assam</li>
  <li>Uttarakhandi from Uttarakhand</li>
  <li>Kashmiri from Kashmir</li>
  <li>Kannada from Karnataka</li>
  <li>Marathi from Maharashtra</li>
</ol> 

Objective of this repository is to provide an
opportunity to researchers to utilize this dataset
and test their algorithms on it. This database can
be used to perform (a) folk songs classification (b)
musical instrument classification (c) language
classification. Except for the first case, classes have
to be manually labeled.

<h3>Current Status</h3>
<ul>
  <li>Audio in 5 Indian Languages</li>
  <li>307 songs and 1807 audio clips</li>
  <br>
  <table>
    <tr>
      <th>Language</th>
      <th>Number of Songs</th>
      <th>Number of Clips</th>
    </tr>
    <tr>
      <td>Assamese</td>
      <td>141</td>
      <td>798</td>
    </tr>
    <tr>
      <td>Uttarakhandi</td>
      <td>29</td>
      <td>174</td>
    </tr>
    <tr>
      <td>Kashmiri</td>
      <td>39</td>
      <td>241</td>
    </tr>
    <tr>
      <td>Kannada</td>
      <td>63</td>
      <td>384</td>
    </tr>
    <tr>
      <td>Marathi</td>
      <td>35</td>
      <td>210</td>
    </tr>
  </table>
</ul>

<h3>Organization</h3>
Files are named in the following format: <i>{Language}{FileNumber}chunk({chunkIndex}).wav</i> Example: Assamese100chunk(0).wav

<h3>Metadata</h3>
<b><i>metadata.py</i></b> contains meta-data regarding the song languages and region of India.

<h3>Included Utilities</h3>
<b><i>featurevector.py</i></b> Used to generate Mean, Median and Standard Deviation of the 19 features of each audio file and convert them into a csv.<br><br>
<b><i>melspectogram.py</i></b> Convert audio data into melSpectogram Visualisation for all categories. <br><br>
<b><i>mfcc_visual.py</i></b> Used to convert the mfcc audio features into Visual representation. <br><br>
<b><i>spectogram.py</i></b> Convert audio data into Spectograms, used often as a preprocessing step.

<h3>Acknowledgement</h3>
This database has been collected through various
publicly available websites and other online
resources.

<h3>Usage</h3>
This database is open for use for any academic or
research purpose.

<h3>Citation</h3>
If you are using this database for your research,
kindly cite following paper
<i>- accepted at FRSM 2020 (citation details will be
available after publication)</i>
