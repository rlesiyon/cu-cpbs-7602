# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] colab_type="text" editable=true id="i_f5u2x9nn6I" slideshow={"slide_type": "slide"} tags=[]
# <left><img width=25% src="img/cu_logo.svg"></left>
#
# # **Lecture 1**: Introduction to Machine Learning
#
# __Milton Pividori__<br>Department of Biomedical Informatics<br>University of Colorado Anschutz Medical Campus

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Welcome to Intro to Big Data in the Biomedical Sciences!
#
# An introduction to "big data" in the context of biomedical sciences must include machine learning (ML).
#
# Machine learning is one of today's most exciting emerging technologies.
#
# In this course, you will learn what machine learning is, some of the most important algorithms in machine learning, and how to apply them to solve problems.
#
# Several lectures are derived from the course *Applied Machine Learning* (Cornell CS5785)

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Agenda
#
# ### Part 1: What is Machine Learning?
# - ML in everyday life (search, spam detection, self-driving cars)
# - Definition and motivation
# - Why ML in biomedical sciences?
#
# ### Part 2: Three Approaches to Machine Learning
# - Supervised, unsupervised, and reinforcement learning
# - AI, ML, Deep Learning, and LLMs
#
# ### Part 3: Logistics
# - Course overview, prerequisites, grading, and policies
# - Software tools and journal club
#
# ### Hands-on Exercise
# - Set up programming environment and run first ML model

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Part 1: What is Machine Learning?
#
# The last few years have produced impressive advances in artificial intelligence.
#
# <table><tr>
#     <td><center><img width=90% src="img/stable-diffusion-example.png"/></center></td>
# <!--     <td><center><img width=100% src="img/midjourney-example.png"/></center></td> -->
#     <td><center><img width=64% src="img/chatgpt.png"/></center></td>
# </tr></table>
#
# Many of these advances have been made possible by _machine learning_.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # ML in Everyday Life: Search Engines
#
# You use machine learning every day when you use a search engine.
#
# <center><img src="img/google.png"/></center>

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # ML in Everyday Life: Spam/Fraud Detection
#
# Machine learning is used in every spam filter, such as in Gmail.
# <br>
# <center><img src="img/spam.png"/></center>

# %% [markdown] slideshow={"slide_type": "-"}
# ML systems are also used by credit card companies and banks to automatically detect fraudulent behavior.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # ML in Everyday Life: Self-Driving Cars
#
# One of the most exciting and cutting-edge uses of machine learning algorithms is in autonomous vehicles.
#
# <center><img width="80%" src="img/waymo.jpg"/></center>

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # A Definition of Machine Learning
#
# In 1959, Arthur Samuel defined machine learning as follows.
#
# > Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.
#
# What do "learn" and "explicitly programmed" mean here? Let's look at an example.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # An Example: Self-Driving Cars
#
# A self-driving car system uses dozens of components that include detection of cars, pedestrians, and other objects.
# <br>
# <center><img width=80% src="img/tesla.jpg"/></center>

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Self-Driving Cars: A Rule-Based Algorithm
#
# One way to build a detection system is to write down rules.
# <left><img width=50% src="img/tesla_zoom.jpg"/></left>
#
# <!-- <table style="border: 1px"><tr>
#     <td><left>One way to build a detection system is to write down rules.</left></td>
#     <td><img src="img/tesla_zoom.jpg"/></td>
# </tr></table> -->

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} tags=[]
# ```python
# # pseudocode example for a rule-based classification system
# object = camera.get_object()
# if object.has_wheels(): # does the object have wheels?
#     if len(object.wheels) == 4: return "Car" # four wheels => car
#     elif len(object.wheels) == 2:
#         if object.seen_from_back():
#             return "Car" # viewed from back, car has 2 wheels
#         else:
#             return "Bicycle" # normally, 2 wheels => bicycle
# return "Unknown" # no wheels? we don't know what it is
# ```

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} tags=[]
# In practice, it's almost impossible for a human to specify all the edge cases.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Self-Driving Cars: An ML Approach
#
# The machine learning approach is to teach a computer how to do detection by showing it many examples of different objects.
#
# <center><img src="img/tesla_data.png"/></center>
#
# No manual programming is needed: the computer learns what defines a pedestrian or a car on its own!

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Revisiting Our Definition of ML
#
# > Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. (Arthur Samuel, 1959.)
#
# This principle can be applied to countless domains:
# medical diagnosis, factory automation, machine translation, and many more!

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Why Machine Learning?
#
# Why is this approach to building software interesting?

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} tags=[]
# * It lets us build practical systems for real-world applications for which other engineering approaches don't work.

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} tags=[]
# * Learning is widely regarded as a key approach towards building general-purpose artificial intelligence systems.

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} tags=[]
# * The science and engineering of machine learning offers insights into human intelligence.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # ... And Why Machine Learning in **Biomedical Sciences**?
#
# * **Enhanced Pattern Recognition**: ML algorithms excel at identifying complex patterns in high-dimensional biomedical data that are beyond human detection.

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} tags=[]
# * **Personalized Medicine**: ML enables the development of models that predict individual patient disease risk and responses to treatments, thereby driving precision medicine and tailored healthcare.

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} tags=[]
# * **Efficient Data Integration**: ML tools can analyze and integrate large, diverse datasets (e.g., genomics, imaging, and clinical data), facilitating holistic insights and novel discoveries.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Example: extracting complex gene networks
#
# <center><img width="60%" src="img/gene_networks.png"/></center>
#
# https://doi.org/10.1146/annurev-biodatasci-103123-095355

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# # Example: extracting complex gene networks
#
# ![image.png](attachment:484625d7-b909-4e9b-987b-04a57dcff07c.png)

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# # Example: extracting complex gene networks
#
# ![image.png](attachment:50738ec9-e1a9-4b68-8ae5-b9280cd095c2.png)

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# # Example: extracting complex gene networks
#
# <center><img width="60%" src="img/lecture01-example_gene_networks01-00.png"/></center>

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# # Example: extracting complex gene networks
#
# <center><img width="60%" src="img/lecture01-example_gene_networks01-01.png"/></center>

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# # Example: extracting complex gene networks
#
# <center><img width="60%" src="img/lecture01-example_gene_networks01-02.png"/></center>

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# # Example: extracting complex gene networks
#
# <center><img width="60%" src="img/lecture01-example_gene_networks01-03.png"/></center>

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# # Example: extracting complex gene networks
#
# <center><img width="60%" src="img/lecture01-example_gene_networks01-04.png"/></center>

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# # Example: extracting complex gene networks
#
# <center><img width="60%" src="img/lecture01-example_gene_networks01-05.png"/></center>

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# <left><img width=25% src="img/cu_logo.svg"></left>
# # Part 2: Three Approaches to Machine Learning
#
# Machine learning is broadly defined as the science of building software that has the ability to learn without being explicitly programmed.
#
# How might we enable machines to learn? Let's look at a few examples.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Supervised Learning
#
# The most common approach to machine learning is supervised learning.
#
#
# <center><img width=90% src="img/supervised_learning_example.png"/></center>
#
# <sub><sup>Image Credit: DataFlair</sup></sub>

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Supervised Learning: Object Detection
# We previously saw an example of supervised learning: object detection.
# <center><img width=70% src="img/tesla_data.png"/></center>
#
# 1. We start by collecting a dataset of labeled objects.
# 2. We train a model to output accurate predictions on this dataset.
# 3. When the model sees new, similar data, it will also be accurate.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Applications of Supervised Learning
#
# Many important applications of machine learning are supervised:
# * Classifying medical images.
# * Translating between pairs of languages.
# * Detecting objects in autonomous driving.
# * Predicting which individuals are at high risk of disease.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Unsupervised Learning
#
# Here, we have a dataset *without* labels.  Our goal is to learn something interesting about the structure of the data.
#
# <center><img width=90% src="img/unsupervised_learning_example.png"/></center>
#
#
# <sub><sup>Image Credit: DataFlair</sup></sub>

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Unsupervised Learning: Text Analysis
#
# In this next example, we have a text containing at least four distinct topics.
#
# <center><img width=60% src="img/lda1.png"/></center>

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# However, we initially do not know what the topics are.
#
# <center><img width=100% src="img/lda3.png"/></center>

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# Unsupervised *topic modeling* algorithms assign each word in a document to a topic and compute topic proportions for each document.
#
# <center><img width=100% src="img/lda2.png"/></center>

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Applications of Unsupervised Learning
#
# Unsupervised learning methods have many other applications:
# * __Recommendation systems__: suggesting movies on Netflix.
# * __Anomaly and outlier detection__: identifying factory components that are likely to fail soon.
# * __Signal processing__: extracting clean human speech from a noisy audio recording.
# * __Disease subtyping__: finding subgroups of individuals with a disease.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Reinforcement Learning
#
# In reinforcement learning, an agent is interacting with the world over time. We teach it good behavior by providing it with rewards.
#
# <center><img width=80% src="img/rl.png"/></center>
#
# <sub><sup>Image by Lily Weng</sup></sub>

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Applications of Reinforcement Learning
#
# Applications of reinforcement learning include:
# * Creating __agents__ that play games such as Chess or Go.
# * __Industrial control__: automatically operating cooling systems in datacenters to use energy more efficiently.
# * __Generative design__ of new drug compounds.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Artificial Intelligence and Deep Learning
#
# Machine learning is closely related to these two fields.
# * AI is about building machines that exhibit intelligence.
# * ML enables machines to learn from experience, a useful tool for AI.
# * Deep learning focuses on a family of learning algorithms loosely inspired by the brain.
# <center><img width="50%" src="img/ai_ml_dl.png"/></center>
#
# <sub><sup>Image [source](https://towardsdatascience.com/understanding-the-difference-between-ai-ml-and-dl-cceb63252a6c).</sup></sub>

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} tags=[]
# **Question:** Where would you place "ChatGPT" here?

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} tags=[]
# * "ChatGPT" or "Claude" are _products_ that use deep learning _models_.

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} tags=[]
# * These are Large Language Models (LLMs) like Llama 4, GPT-5, Claude Sonnet 3.5, etc.

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# **Question:** Are LLMs supervised, unsupervised, RL, ...?

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} tags=[]
# * LLMs don't fit neatly into any of those; they are **hybrid learners**.

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} tags=[]
# * They are **autoregressive Transformer neural networks**:
#   * predict the next token
#   * use the Transformer architecture (attention, etc.)
#   * trained on massive text corpora

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} tags=[]
# * There are multiple stages in their **training**:
#   * pretraining (provides language understanding)
#   * supervised fine-tuning, RLHF, etc. (makes them more helpful, safe, etc.).

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} tags=[]
# > They are **self-supervised models** with supervised and RL alignment.

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# **Question:** And what about PLIER, VAE, ...?
#
# <center><img width="50%" src="img/lecture01-example_gene_networks01-05.png"/></center>

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} tags=[]
# > A vanilla VAE (middle) is an **unsupervised model**
# > 
# > PLIER and interpretable VAE are **semi-supervised models**

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# <left><img width=25% src="img/cu_logo.svg"></left>
# # Part 3: Logistics
#
# We conclude the lecture with the logistical aspects of the course.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # What Is the Course About?
#
# This course introduces machine learning algorithms from a very practical perspective.
# * __Algorithms__: We cover a *broad* set of ML algorithms.
# * __Applications__: We'll explore scientific articles to learn how ML is applied in biomedical sciences.
#
# We won't focus deeply on the mathematical foundations of algorithms nor implement them from scratch.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Course Contents
#
# Some of the most important sets of topics we will cover include:
# * __Supervised Learning__: Regression, classification.
# * __Unsupervised Learning__: Clustering, dimensionality reduction.
# * __Statistical inference__: multiple testing, permutations.
# * __Applying ML__: Evaluation, overfitting, regularization.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Prerequisites: Is This Course For You?
#
# The course is intended for students from:
#
# * Computational Biosciences,
# * Neuroscience,
# * Human Medical Genetics and Genomics,
# * and related programs where student research will be primarily computational.

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} tags=[]
# The main requirements for this course are:
#
# * __Programming__: At least two years of experience in either R, Python, or equivalent.
# * __Computational coursework__.
#
# This course does not assume any prior ML experience.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Logistics
#
# **When:** We meet Mon-Fri 10:00 a.m. - 12:00 p.m. (Nov 21 - Dec 12, 2025)
# <br />
# **Where:** Academic Building, 2nd Floor – Room L15-2101 (Department Conference Room)
#
# * __Class Webpage__: https://ucdenver.instructure.com/courses/566796
# * **Milton Pividori (Instructor)**:
#   * Email: milton.pividori@cuanschutz.edu
#   * Office location: Anschutz Health Sciences Building (AHSB) P12-7071
#   * Office hours: by appointment
# * **Aishwarya Mandava (Teaching Assistant)**:
#   * Email: aishwarya.mandava@cuanschutz.edu
#   * Office location: AHSB P12-6003
#   * Office hours: Mondays, 3:00pm - 4:30pm

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# Please, reach out to us **via email**! We might not answer in a timely manner in Canvas.
#
# You all should be part of an **Outlook group**.

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# We will use Canvas for the course.
#
# * __Assignments__: download and submit all assignments there
#     * Click the "Assignments" tab in Canvas
# * __Canvas Discussions__: Use the forum to ask us questions online
#     * Click the "Discussions" tab in Canvas
# * __Canvas Announcements__: Make sure to regularly watch for updates
#     * Click the "Announcements" tab in Canvas

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Lecture Slides
#
# The slides for each lecture will be available online on Canvas before the lecture.
#
# * We will also share lecture notes in HTML and PDF format
# * Look out for links and announcements on Canvas

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Grading
#
# * See syllabus for criteria.
# * We will have three assignments and three journal clubs.
#   * __Three Assignments__: 75% - Conceptual questions & programming problems.
#   * __Three Journal Clubs__: 25% - Participation
# * __Assignments are due a week after__ they are released, at 5 pm MT.
#   * You have one late day with 10% penalty in grade.
#   * Submissions later than this need to be communicated to instructors.
# * __See syllabus for class attendance policies__:
#   * In-person and on-time attendance is required.
#   * Arriving late.
#   * Chronic lateness.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Policy on Generative AI
#
# * You are allowed but discouraged to use Generative AI.
# * If you choose to use it:
#   * Do it intelligently: **don't harm your learning and creativity**.
#   * You must add a statement explaining how you used it.

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} tags=[]
# <div><img width="60%" src="img/chatbots_science.png"/></div>
#
# * <u>URL</u>: https://doi.org/10.1038/d41586-024-02630-z
# * <u>PDF</u> available in Canvas

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Software You Will Use
#
# You will use **Python** and **R** and popular machine learning libraries such as:
#
#   * `scikit-learn`. It implements most classical machine learning algorithms.
#   * `numpy`, `pandas`, `dplyr`. Linear algebra and data processing libraries used to implement algorithms from scratch.
#   * `matplotlib`, `seaborn`, `ggplot2`. For plotting.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Announcements
#
# For next journal club, we'll be reading this article:
#
# ![image.png](attachment:fb6be750-e671-4eee-9857-a8df3375723e.png)
#
# **Link**: https://doi.org/10.1016/S2213-8587(18)30051-2 (PDF in Canvas)

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# ## Modality for journal club
#
# ### Before
#   * Everyone reads the entire paper
#   * Everyone submits a *brief document*:
#     * Write brief responses (3-5 sentences each) to 3-4 guiding questions
#     * You pick the questions; make sure to focus on different aspects
#     * Email Milton with subject *"CPBS 7602 - Journal Club 01"*

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# ## Modality for journal club
#
# ### During
# #### Step 1: each student is assigned a theme or question cluster
# * <u>Motivation analyst</u> (goals, motivation, take-home message)
#   * Example: Summarize study aims and big-picture context
# * <u>Methods detective</u> (three things learned, unclear aspects)
#   * Example: Highlight a technical point and explain or question it
# * <u>Data skeptic</u> (figures, tables, data interpretation)
#   * Example: Challenge whether data support the conclusions
# * <u>Supplemental scout</u> (supplemental data, source code, missing information)
#   * Example: Present 1 interesting supplemental figure/table
# * <u>Future thinker</u> (orthogonal analyses, implications for own work)
#   * Example: Suggest extensions or new ideas

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# ## Modality for journal club
#
# ### During
# #### Step 2: discussion
# * <u>Opening summary</u> (0-5 min): a student moderator summarizes the paper
# * <u>Motivation & Methods</u> (5-25 min): "Motivation Analyst" and "Methods Detective" groups lead discussion
# * <u>Data & Interpretation</u> (25-45 min): "Data Skeptic" and "Supplemental Scout" discuss results
# * <u>Extensions & Reflection</u> (45-60 min): "Future Thinker" group presents ideas
# * <u>Wrap-up & synthesis</u> (60-70 min): Summarize key takeaways

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# ## Journal club: 10 questions to discuss
#
# <div style="font-size: small;">
#   <ol>
#     <li>What are the goals of the study? What is the motivation?</li>
#     <li>In your own words, what is the take home message of this paper? i.e. what do you hope to remember about this work?</li>
#     <li>List 3 things you learned that you didn't know before. These can be about background, methods, or anything else.</li>
#     <li>List the things you didn't understand about the paper (so you can follow up).</li>
#     <li>What's the most important or interesting thing you saw in the supplemental data?</li>
#     <li>Are the data presented in a manner that allows you to interpret them? Is any relevant information missing? Are the figures clear and/or quantified?</li>
#     <li>If you look at the data (figures and tables) alone, do you reach the same conclusions as the authors do? Do the data suggest an alternative explanation?</li>
#     <li>Suggest at least one orthogonal analysis not in the paper that might strengthen the results or might have been included instead.</li>
#     <li>Does this paper make you think differently about your own project or experiments? If so, make note.</li>
#     <li>What topic did this paper make you want to read more about?</li>
#   </ol>
# </div>
#

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# ### Again, Welcome to Intro to Big Data in the Biomedical Sciences!
# <left><img width=25% src="img/cu_logo.svg"></left>

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Executable Course Materials
#
# The core materials for this course (including the slides) are created using Jupyter notebooks.
# * We are going to embed and execute code directly in the slides and use that to demonstrate algorithms.
# * These slides can be downloaded locally and all the code can be reproduced.

# %% editable=true slideshow={"slide_type": "fragment"} tags=[]
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

plt.rcParams["figure.figsize"] = [12, 4]

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# We can use these libraries to load a simple dataset of handwritten digits.

# %% editable=true slideshow={"slide_type": "fragment"} tags=[]
# https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
# load the digits dataset
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images.
_, axes = plt.subplots(1, 4)
images_and_labels = list(zip(digits.images, digits.target))
for ax, (image, label) in zip(axes, images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Label: %i" % label)

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# We can now load and train this algorithm inside the slides.

# %% editable=true slideshow={"slide_type": "fragment"} tags=[]
np.random.seed(0)
# To apply a classifier on this data, we need to flatten the image, to
# turn the data into a (samples, feature) matrix:
data = digits.images.reshape((len(digits.images), -1))
n_samples = data.shape[0]
print(data)
# create a small neural network model
from sklearn.model_selection import train_test_split  # noqa E402
from sklearn.neural_network import MLPClassifier  # noqa E402

classifier = MLPClassifier(alpha=1e-3)

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# We learn the digits on the first half of the digits
classifier.fit(X_train, y_train)

# Now predict the value of the digit on the second half:
predicted = classifier.predict(X_test)

# %% [markdown]
# Multi class classification

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# We can now visualize the results.

# %% editable=true slideshow={"slide_type": "fragment"} tags=[]
_, axes = plt.subplots(1, 4)
images_and_predictions = list(zip(digits.images[n_samples // 2 :], predicted))
for ax, (image, prediction) in zip(axes, images_and_predictions[:4]):
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Prediction: %i" % prediction)

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # **Hands-on**: creating a programming environment
#
# * Create a personal GitHub repository named `cu-cpbs-7602`
#   * Here you'll store your exercises and assignments.
# * Create a conda environment to run your code (see next slide).
#   * You'll find an `environment.yml` file on Canvas.
# * Download and run this lecture's notebook `lecture01-introduction.ipynb`:
#   * Notebook is on Canvas.
#   * Make sure you can run the code under "Executable Course Materials"
#   * At the end of the notebook, add a "Markdown" cell, and using the concepts learned during the lecture, describe what kind of model you just ran.

# %%

# %%

# %%

# %%

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# ## Creating a conda environment
# 1. Install [Miniforge](https://github.com/conda-forge/miniforge).
# 1. Create the environment for this course:
#    ```bash
#    # create conda environment
#    conda env create --name cu-cpbs-7602 --file environment.yml
#    
#    # activate environment in your terminal session
#    conda activate cu-cpbs-7602
#    
#    # run Jupyter Lab server (it will automatically open your browser)
#    jupyter lab
#    ```

# %% [markdown] editable=true slideshow={"slide_type": "subslide"} tags=[]
# ## Tips
# * Some suggested packages that make life easier (already included in `environment.yml`):
#     * `jupyterlab`
#     * `jupytext`
#       * There is a file `jupytext.toml` on Canvas.
#       * You can copy it to the folder where you have your nootebooks.
#       * It will create a paired `.py` file that is easier to `diff`
# * *Tip:* Throughout the course, check out the `scikit-learn` [user guide](https://scikit-learn.org/stable/user_guide.html) and [API](https://scikit-learn.org/stable/api/index.html) (for specific models).
#     * For this example, the documentation for the [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) might be helpful.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} tags=[]
# # Anonymous Feedback On This Lecture
#
# <center>
#     <img width=60% src="img/feedback_form_qr.png">
# </center>
#
# * **Link:** https://forms.office.com/r/prXY35CNuz
# * **Lecture number:** 1
# * **Lecture topic:** intro

# %% editable=true slideshow={"slide_type": "skip"} tags=["remove_cell"]
