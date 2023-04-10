> [What & Why]: motivate the need for the product and outline the objectives and key results.

 
 # Background - Customer-centric approach 

Customer : machine learning developers and researchers.
> Profile of the customer we want to address

Goal: stay up-to-date on ML content for work, knowledge, etc.
> Main goal for the customer

Pains: too much uncategorized content scattered around the internet.
> Obstacles in the way of the customer achieving the goal

Gains: a central location with categorized content from trusted 3rd party sources.
> What would make the job easier for the customer?

# Value proposition - Product-centric approach

Product: service that discovers and categorizes ML content from popular sources.    
>what needs to be build to help the customer reach their goal?

Alleviates: timely display categorized content for customers to discover.
> how will the product reduce pains?

Advantages: customers only have to visit our product to stay up-to-date.
>how will the product create gains?

# Key objectives & Solution

Describe the solution required to meet our objectives, including it's core features, integration, alternatives, constraints and what's out-of-scope.

Develop a model that can classify the incoming content so that it can be organized by category on our platform.

Core features:

ML service that will predict the correct categories for incoming content. [OUR FOCUS]
user feedback process for incorrectly classified content.
workflows to categorize content that the service was incorrect about or not as confident in.
duplicate screening for content that already exists on the platform.
Integrations:

categorized content will be sent to the UI service to be displayed.
classification feedback from users will sent to labeling workflows.
Alternatives:

allow users to add content manually (bottleneck).
Constraints:

maintain low latency (>100ms) when classifying incoming content. [Latency]
only recommend tags from our list of approved tags. [Security]
avoid duplicate content from being added to the platform. [UI/UX]
Out-of-scope:

identify relevant tags beyond our approved list of tags.
using full-text HTML from content links to aid in classification.
interpretability for why we recommend certain tags.
identifying multiple categories (see dataset section for details).


# Feasibility

How feasible is our solution and do we have the required resources to deliver it (data, $, team, etc.)?