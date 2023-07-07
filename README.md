# HPE-DA
 2023 Business Analytics Student Competition! HPE UTA
# Media Mix Modelling using Genetic Algorithms and Multivariate Regression

The advertising industry is vast… The numbers say it all. Statista has projected that the industry to reach a value of $800 billion by 2027. The two major players are Facebook and Google. Spending on Facebook ads every year is getting close to $500 billion, with a daily spend of $1.36B dollars. Spending on Google ads is over $400M every day. Big numbers! The powers of Computation modeling and AI definitely aid our purpose. Utilizing data science and machine learning is crucial for making informed and wise decisions.

**Problem Statement**

1. $1M of budget for Ad Campaign
2. Distributed across 3 Digital Channels and 5 Audience types 
3. Optimize the Budget while Maximizing Digital Engagement
4. Business Constraint - Minimum of 2% and a maximum of 50% Total Spend 

**Data**

1. 5 Ad Campaigns consisting of $ Spend for 375000 ads across different Channels, Audiences, and Content for 90 days
2. Sucess Metrics - Impressions, Clicks, Video Views, Web Visits, Social Likes
3. Digital Engagement - Click-through Rate = Clicks/Impressions

**Formulation**

                𝐼𝑚𝑝𝑟𝑒𝑠𝑠𝑖𝑜𝑛𝑠= 𝛽_0+ 𝛽_1 𝐶ℎ𝑎𝑛𝑛𝑒𝑙+ 𝛽_2 𝐴𝑢𝑑𝑖𝑒𝑛𝑐𝑒+ 𝛽_3 𝑆𝑝𝑒𝑛𝑑
                
                𝐶𝑙𝑖𝑐𝑘𝑠= 𝛽_0+ 𝛽_1 𝐶ℎ𝑎𝑛𝑛𝑒𝑙+ 𝛽_2 𝐴𝑢𝑑𝑖𝑒𝑛𝑐𝑒+ 𝛽_3 𝑆𝑝𝑒𝑛𝑑
                
                𝐶𝑇𝑅_𝑖𝑗=(𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑒𝑑 𝐶𝑙𝑖𝑐𝑘𝑠)/(𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑒𝑑 𝐼𝑚𝑝𝑟𝑒𝑠𝑠𝑖𝑜𝑛𝑠)=𝑓(𝑆𝑝𝑒𝑛𝑑_𝑖𝑗 )/𝑔(𝑆𝑝𝑒𝑛𝑑_𝑖𝑗 ) =ℎ(𝑆𝑝𝑒𝑛𝑑_𝑖𝑗 )








