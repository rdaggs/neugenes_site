## Overview

I want to build an interactive platform to visualize neural and genetic data from deep-learning alignment models from my own Neuroscience project. 
It will allow users to dump in their full brain datasets, and be served a digestible visual result after being processed. 


## Data Model
Users can have each slice of their brain dataset receive equal attention 

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


## Research Topics

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


##  References Used



1. [Express.js Documentation](https://expressjs.com/) - Basic Express application structure and middleware setup
2. [Mongoose Documentation](https://mongoosejs.com/docs/guide.html) - Schema design and MongoDB connection
3. [Express Handlebars](https://github.com/express-handlebars/express-handlebars) - View engine configuration


