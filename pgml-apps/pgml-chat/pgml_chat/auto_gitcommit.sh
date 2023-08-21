#!/bin/bash

# Function to check if a branch exists locally
branch_exists() {
    git show-ref --verify --quiet refs/heads/$1
}

# Check git status
echo "Git Status:"
git status

# Add all changes
echo "Adding all changes..."
git add .

# Commit changes
read -p "Enter commit message: " commit_message
git commit -m "$commit_message"

# Ask for branch name and check if it exists
read -p "Enter branch name to push (e.g., main): " branch
while ! branch_exists $branch; do
    echo "Branch named $branch doesn't exist locally."
    read -p "Do you want to create it? (yes/no) " decision
    if [[ $decision == "yes" ]]; then
        git checkout -b $branch
    else
        read -p "Enter a different branch name to push or 'exit' to quit: " branch
        if [[ $branch == "exit" ]]; then
            exit 0
        fi
    fi
done

# Push to the branch
git push origin $branch
if [ $? -ne 0 ]; then
    echo "Error: Failed to push to $branch. Check your remote repository and network connection."
    exit 1
fi

echo "Changes pushed to $branch successfully!"

