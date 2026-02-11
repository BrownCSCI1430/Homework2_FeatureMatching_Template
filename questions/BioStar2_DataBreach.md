# Report: Data Breach in Biometric Security Platform Affecting Millions of Users

**By vpnMentor Research Team** — *Cybersecurity and Research Lab*
*Published on August 14, 2019*
https://www.vpnmentor.com/blog/cybersecurity/report-biostar2-leak/

---

Led by internet privacy researchers Noam Rotem and Ran Locar, [vpnMentor's](https://www.vpnmentor.com/) team recently discovered a huge data breach in security platform BioStar 2.

BioStar 2 is a web-based biometric security smart lock platform. A centralized application, it allows admins to control access to secure areas of facilities, manage user permissions, integrate with 3rd party security apps, and record activity logs.

As part of the biometric software, BioStar 2 uses facial recognition and fingerprinting technology to identify users.

The app is built by Suprema, one of the world's top 50 security manufacturers, with the highest market share in biometric access control in the EMEA region. Suprema recently partnered with Nedap to integrate BioStar 2 into their AEOS access control system.

AEOS is used by over 5,700 organizations in 83 countries, including some of the biggest multinational businesses, many small local businesses, governments, banks, and even the UK Metropolitan Police.

The data leaked in the breach is of a highly sensitive nature. It includes detailed personal information of employees and unencrypted usernames and passwords, giving hackers access to user accounts and permissions at facilities using BioStar 2. Malicious agents could use this to hack into secure facilities and manipulate their security protocols for criminal activities.

This is a huge leak that endangers both the businesses and organizations involved, as well as their employees. Our team was able to access over 1 million fingerprint records, as well as facial recognition information. Combined with the personal details, usernames, and passwords, the potential for criminal activity and fraud is massive.

Once stolen, fingerprint and facial recognition information cannot be retrieved. An individual will potentially be affected for the rest of their lives.

## Timeline of Discovery and Owner Reaction

After we discovered the breach in BioStar 2's database, we contacted the company alerting them of our findings.

However, we found BioStar 2 generally very uncooperative throughout this process. Our team made numerous attempts to contact the company over email, to no avail. Eventually, we decided to reach out to BioStar 2's offices by phone. Again, the company was largely unresponsive.

Upon speaking to a member of their German team, we received a mumbled reply that "we don't speak to vpnMentor", before the phone was suddenly hung up. This suggests they were aware of us, and our attempts to resolve the issue.

We also tried to contact BioStar 2's GDPR compliance officer but received no reply.

Eventually, after speaking to the more cooperative French branch over the phone, steps were taken by the company to close the breach.

- Date discovered: 5th August 2019
- Date vendors contacted: 7th August 2019
- Date of Action: 13th August, the breach was closed

## Example of Entries in the Database

As a centralized security solution, BioStar 2's database contained almost every kind of sensitive data available.

This could be used in a wide range of criminal activities that would be disastrous for both the businesses and organizations affected, as well as their employees or clients.

Our team was able to access over 27.8 million records, a total of 23 gigabytes of data, which included the following information:

- Access to client admin panels, dashboards, back end controls, and permissions
- Fingerprint data
- Facial recognition information and images of users
- Unencrypted usernames, passwords, and user IDs
- Records of entry and exit to secure areas
- Employee records including start dates
- Employee security levels and clearances
- Personal details, including employee home address and emails
- Businesses' employee structures and hierarchies
- Mobile device and OS information

One of the more surprising aspects of this leak was how unsecured the account passwords we accessed were. Plenty of accounts had ridiculously simple passwords, like "Password" and "abcd1234". It's difficult to imagine that people still don't realize how easy this makes it for a hacker to access their account.

Of course, many users did create more complicated and effective passwords that normally would be difficult to discover or decrypt. However, we were easily able to view passwords across the BioStar 2 database, as they were stored as plain text files, instead of being securely hashed.

The range of businesses affected by the leak varied widely in size, location, industry, and users. Some examples of businesses whose information we were able to access and view worldwide included:

USA

- [Union Member House](https://www.unionmemberhouse.com/about) - Coworking space and social club with 7,000 users.
- [Lits Link](https://litslink.com/) - Software development consultancy.
- [Phoenix Medical](http://phoenixmedicalinc.com/) - Medical products manufacturer.

Indonesia

- [Uptown](https://uptown.id/) - Jakarta-based coworking space with 123 users.

India and Sri Lanka

- [Power World Gyms](https://powerworldgyms.com/) - High-class gym franchise with branches across both countries. We accessed 113,796 user records and their fingerprints.

United Kingdom

- [Associated Polymer Resources](http://www.assoc-polymer-resources.co.uk/) - Plastics recycling specialists.
- [Tile Mountain](https://www.tilemountain.co.uk/) - Home decor and DIY supplier.
- [Farla Medical](https://www.farlamedical.co.uk/) - Medical supply store.

UAE

- [Global Village](https://www.globalvillage.ae/en/) - An annual cultural festival, with access to 15,000 fingerprints.
- [IFFCO](http://iffco.com/) - Consumer food products group.

Finland

- [Euro Park](https://europark.fi/) - Car parking space developer with sites across Finland.

Turkey

- [Ostim](http://www.ostim.org.tr/) - Industrial zone construction developer.

Japan

- [Inspired.Lab](https://inspiredlab.jp/) - Coworking and design space in Chiyoda City, Tokyo.

Belgium

- [Adecco Staffing](https://www.adecco.com/) - We found approximately 2,000 fingerprints connected to the staffing and human resources giant.

Germany

- [Identbase](https://identbase.de/) - Data belonging to this supplier of commercial ID and access card printing technology was also found in the exposed database.

Maybe the biggest concern in this leak is its size. BioStar 2's users are spread around the world, with potential future users including governments, banks, universities, defense contractors, police, and multinational businesses.

The platform has over 1.5 million worldwide installations, and all of these could be vulnerable to this leak. The total number of people affected could be in the tens of millions.

## Data Breach Impact

Facial recognition and fingerprint information cannot be changed. Once they are stolen, it can't be undone. The unsecured manner in which BioStar 2 stores this information is worrying, considering its importance, and the fact that BioStar 2 is built by a security company.

Instead of saving a hash of the fingerprint (that can't be reverse-engineered) they are saving people's actual fingerprints that can be copied for malicious purposes.

Putting all the data found in the leak together, criminals of all kinds could use this information for varied illegal and dangerous activities.

### Account Takeovers and Security Breaches

With this leak, criminal hackers have complete access to admin accounts on BioStar 2. They can use this to take over a high-level account with complete user permissions and security clearances, and make changes to the security settings in an entire network.

Not only can they change user permissions and lock people out of certain areas, but they can also create new user accounts - complete with facial recognition and fingerprints - to give themselves access to secure areas within a building or facility.

Furthermore, hackers can change the fingerprints of existing accounts to their own and hijack a user account to access restricted areas undetected. Hackers and other criminals could potentially create libraries of fingerprints to be used any time they want to enter somewhere without being detected.

This provides a hacker and their team open access to all restricted areas protected with BioStar 2. They also have access to activity logs, so they can delete or alter the data to hide their activities.

As a result, a hacked building's entire security infrastructure becomes useless. Anybody with this data will have free movement to go anywhere they choose, undetected.

### Robbery and Fraud

The most obvious danger of giving a hacker or criminal complete access to a secure building is theft. They can use this database to quite literally walk into a room and take anything of value.

This is true no matter the nature of the building, whether it's a small-town gym or a government office.

The leak also gives hackers access to otherwise closed networks that they might not be able to reach from outside a building. With this, they can steal valuable information, plant viruses, monitor and exploit systems, and much more.

### Identity Theft and Fraud

The BioStar 2 leak contained huge amounts of individual personal details alongside the users' names, fingerprints, and images. This included employment records, email addresses, and home addresses.

Aside from the security concerns for businesses affected, employees and clients can now be targeted for fraud and other crimes.

The same personal details can also be used to craft effective phishing campaigns. A phishing campaign is the use of imitation emails to trick victims into clicking a link embedding with malware or providing information that can be used to steal from them. With the personal and professional details available in this leak, creating effective phishing campaigns would not be difficult.

The BioStar 2 data gives cybercriminals a solid foundation to exploit users for illegal financial gain. They can also sell the information - including fingerprints - on the dark web to other criminals or malicious agents. This could lead to many untraceable, incriminating activities committed with the data of innocent BioStar 2 users.

### Blackmail and Extortion

Targeting certain employees for blackmail or extortion based on their access permissions in a business is a popular tactic used by criminals around the world. It allows a hacker to gain access to valuable information or assets without putting themselves in physical danger.

The breach in BioStar 2's database allows hackers to view individual security clearances within an organization and target high-level individuals for blackmail and extortion based on this.

Using the personal details available, they can make their threats very effective, by accessing private information and exploiting personal vulnerabilities like family or relationships. This puts employees of the affected BioStar 2 clients under a great deal of potential danger.

### Using Stolen Fingerprints

The use of biometric security like fingerprints is a recent development. As such, the full potential danger in having your fingerprints stolen is still unknown.

However, the important thing to remember is that once it's stolen, unlike passwords, your fingerprint can't be changed.

This makes fingerprint data theft even more concerning. Fingerprints are replacing typed passwords on many consumer items, like phones. Most fingerprint scanners on consumer goods are unencrypted, so when a hacker develops technology to replicate your fingerprint, they will gain access to all the private information such as messages, photos, and payment methods stored on your device.

This is just one potential issue of many.

For BioStar 2, one of the biggest issues right now is reputational. We're concerned that a security company has failed to fully protect its clients.

In the hands of criminal hackers, all this data could all have been downloaded and saved for later use in a variety of crimes.

## Advice from the Experts

This leak could have been easily avoided, had the makers of BioStar 2 taken some basic security precautions. While the information we found could still have made it into the hands of criminal hackers, we suggest the following to BioStar 2 and Suprema:

1. Secure your servers with better protection measures.
2. Don't save the actual fingerprints of users. Save a hash version that can't be reverse-engineered.
3. Implement proper access rules on your databases.
4. Never leave a system that doesn't require authentication open to the internet.

Instead of saving a hash of the fingerprint (that cannot be reverse-engineered) they are saving the actual fingerprint that can then be used to create a copy for malicious purposes.

### Advice to BioStar 2 Clients

If your business or organization is using BioStar 2 and you're concerned you've been affected by this data breach, we suggest you contact Suprema for more details.

We also suggest changing the password to your BioStar 2 dashboard immediately and notifying staff to change their personal passwords.

Additionally, we suggest creating a guide or sharing tools with your staff to help them [generate secure passwords](https://www.vpnmentor.com/tools/secure-password-generator/). There are plenty of [online password meters](https://www.vpnmentor.com/tools/passwordmeter/) available to ensure that they are better protected.

For an in-depth guide on how to protect your business online, check out [how to secure your website](https://www.vpnmentor.com/blog/internet-safety/how-to-secure-website-database/) and online database from hackers.

### Advice to Users

If your employer, or a business you're a customer of, uses BioStar 2, your personal information, fingerprints, and facial recognition data may have been leaked.

You should notify the business or employer of your concerns and ensure they're aware of the data leak.

If you're concerned about data vulnerabilities in general, read our [complete guide to online privacy](https://www.vpnmentor.com/blog/research/ultimate-guide-online-privacy/). It shows you the many ways you can be targeted by cybercriminals, and the steps you can take to stay safe.

## How and Why We Discovered the Breach

vpnMentor's research team found the breach through a huge web-mapping project. Headed by Noam and Ran, the team scans ports looking for familiar IP blocks. They use these blocks to find holes in a company's web system. Once these holes are found, the team looks for vulnerabilities that would lead them to a data breach.

Our team found out that substantial sections of BioStar 2's database are not safeguarded and remain largely unencrypted. The enterprise employs an Elasticsearch database, which is typically not intended for URL utilization. Nonetheless, we managed to gain access through a browser and were able to modify the URL search parameters, which led to the disclosure of a significant quantity of data.

Using their expertise, they also examined the database to confirm its identity.

As [ethical hackers](https://www.vpnmentor.com/blog/research/top-5-places-learn-ethical-hacking-online/), we are obliged to reach out to websites when we discover security flaws. This is especially true when a company's data breach affects so many people and contains such sensitive data.

However, these ethics also mean we carry a responsibility to the public. BioStar 2 customers and their employees must be aware of the risks they take when using technology that makes so little effort to protect their users.

## About Us and Previous Reports

[vpnMentor](https://www.vpnmentor.com/) is the world's largest VPN review website. Our research lab is a pro bono service that strives to help the online community defend itself against cyber threats while educating organizations on protecting their users' data.

We recently discovered a huge [data breach impacting 80 million US households](https://www.vpnmentor.com/blog/cybersecurity/report-millions-homes-exposed/). We also revealed that [Gearbest experienced a massive data breach](https://www.vpnmentor.com/blog/cybersecurity/gearbest-hack/). You may also want to read our [VPN Leak Report](https://www.vpnmentor.com/blog/cybersecurity/vpn-leaks-found-3-major-vpns-3-tested/) and [Data Privacy Stats Report](https://www.vpnmentor.com/blog/research/vpn-use-data-privacy-stats/).