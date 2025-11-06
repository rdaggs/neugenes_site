The content below is an example project proposal / requirements document. Replace the text below the lines marked "__TODO__" with details specific to your project. Remove the "TODO" lines.

(__TODO__: your project name)

# Shoppy Shoperson 

## Overview

I want to build a web app that hosts my own Neuroscience project. 
It will allow users to dump in their full brain datasets, and be served a digestible visual result after being processed. 


## Data Model
Users can have each slice of their brain dataset receive equal attention 

(__TODO__: sample documents)

An Example User:

```javascript
{
  username: String,
  email: String,
  passwordHash: String,
  bio: String,
  profilePicture: String, // URL or path
  createdAt: Date,
  followers: [ObjectId] // references to other Users
}
```

An Example List with Embedded Items:

```javascript
{
  title: String,
  description: String,
  imageUrl: String, // or array of URLs for multiple images
  medium: String, // e.g., "Digital Art", "Oil Painting", "Photography"
  tags: [String], // e.g., ["portrait", "watercolor", "nature"]
  collection: String, // e.g., "My Best Work 2024"
  dateCreated: Date,
  uploadedAt: Date,
  artist: ObjectId, // reference to User
  likes: Number,
  likedBy: [ObjectId], // references to Users who liked it
  comments: [ObjectId] // references to Comment documents
}
```


## [Link to Commented First Draft Schema](db.mjs) 

db.mjs

## Wireframes

(__TODO__: wireframes for all of the pages on your site; they can be as simple as photos of drawings or you can use a tool like Balsamiq, Omnigraffle, etc.)

/list/create - page for creating a new shopping list

![list create](documentation/list-create.png)

/list - page for showing all shopping lists

![list](documentation/list.png)

/list/slug - page for showing specific shopping list

![list](documentation/list-slug.png)

## Site map

```
Home Page
├── Browse Artists
│   └── Artist Profile Page
│       └── Individual Artwork Page
│           └── Comment on Artwork (requires login)
├── Login/Register
├── Search Results
└── User Dashboard (authenticated users only)
    ├── My Profile (view as visitors see it)
    ├── Upload New Artwork
    ├── Edit Artwork
    └── Manage Collections

## User Stories or Use Cases

(__TODO__: write out how your application will be used through [user stories](http://en.wikipedia.org/wiki/User_story#Format) and / or [use cases](https://en.wikipedia.org/wiki/Use_case))
1. **As a musician**, I want to upload my sounds with descriptions so that potential clients can see my portfolio
2. **As a photographer**, I want to set my photos into collections (e.g., "Weddings," "Nature") so that visitors can easily browse specific types of work
3. **As a design student**, I want to tag my projects with keywords so that they're discoverable when people want to hire me
4. **As an art fan**, I want to browse different artists' portfolios so that I can find inspiration
5. **As a visitor**, I want to comment on artwork so that I can give feedback to the artist
6. **As an artist**, I want to see how many likes my artwork gets so that I know which pieces resonate with viewers
7. **As a user**, I want to edit or delete my artwork so that I can keep my portfolio up-to-date
8. **As an illustrator**, I want to display my social media links on my profile so that people can follow me on other platforms
9. **As a visitor**, I want to search for specific styles or mediums (e.g., "watercolor portraits") so that I can find relevant artwork
10. **As an artist**, I want a dashboard to manage all my content in one place so that portfolio maintenance is efficient


## Research Topics

(__TODO__: the research topics that you're planning on working on along with their point values... and the total points of research topics listed)


Integrate user authentication with Passport.js
* Same as hw6

Cloud-based image upload and storage with Cloudinary
* I'm going to use Cloudinary's API for image upload, storage, and delivery
Artists need to upload high-quality images of their work; storing in MongoDB is inefficient
* I'm going to use Cloudinary's API for audio file upload, storage, and delivery. Musicians need to upload high-quality audio files of their work; storing large audio files in MongoDB is inefficient and impractical. Cloudinary supports various audio formats (MP3, WAV, OGG, FLAC) and provides automatic transcoding, streaming capabilities, and CDN delivery for fast playback


CSS framework - Tailwind CSS
  * I'm going to use Tailwind CSS for styling the application. Tailwind's utility-first approach will help create a clean, modern interface quickly

VUE.js 
  * For my artist portfolio. Dynamic artwork galleries - Filter/sort without page reloads. Real-time like counters - Update instantly when users like artwork. Interactive search - Live filtering as users type

* (5 points) Integrate user authentication
    * I'm going to be using passport for user authentication
    * And account has been made for testing; I'll email you the password
    * see <code>cs.nyu.edu/~jversoza/ait-final/register</code> for register page
    * see <code>cs.nyu.edu/~jversoza/ait-final/login</code> for login page
* (4 points) Perform client side form validation using a JavaScript library
    * see <code>cs.nyu.edu/~jversoza/ait-final/my-form</code>
    * if you put in a number that's greater than 5, an error message will appear in the dom
* (5 points) vue.js
    * used vue.js as the frontend framework; it's a challenging library to learn, so I've assigned it 5 points

10 points total out of 8 required points (___TODO__: addtional points will __not__ count for extra credit)


## [Link to Initial Main Project File](app.mjs) 

(__TODO__: create a skeleton Express application with a package.json, app.mjs, views folder, etc. ... and link to your initial app.mjs)

## Annotations / References Used

(__TODO__: list any tutorials/references/etc. that you've based your code off of)

1. [Express.js Documentation](https://expressjs.com/) - Basic Express application structure and middleware setup
2. [Mongoose Documentation](https://mongoosejs.com/docs/guide.html) - Schema design and MongoDB connection
3. [Express Handlebars](https://github.com/express-handlebars/express-handlebars) - View engine configuration


