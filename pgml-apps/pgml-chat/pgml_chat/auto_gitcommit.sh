#!/bin/bash

# Check git status
echo "Git Status:"
git status

# Add all changes
echo "Adding all changes..."
git add .

# Commit changes
read -p "Enter commit message: " commit_message
git commit -m "$commit_message"

# Push to the branch
read -p "Enter branch name to push (e.g., main): " branch
git push origin $branch

echo "Changes pushed to $branch successfully!"

