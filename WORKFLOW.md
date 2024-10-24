# Capstone Git and Github Usage

## Terms and Definitions

`Origin` :  Remote Repository on Github\
`Upstream`: Another name for `origin`\
`Master` :    The branch `main`, either local or remote

## Using Git and Github Collaboratively

I recommend using VS Code as the primary editor as there are extensions that are very useful including git and python. It will allow you to sign into Github, and to update local branches using the git bash terminal within VS Code.

Once milestones have been reached or certain features are going to be added to the `Master` branch, a pull request for that feature branch can be submitted through Github. This will send a notification to me and allow me to review the code before merging it with `main`.

### Before Working

Before starting to do anything locally on your computer, ensure that your local repository is up to date with the remote. This can be done via a `pull` or `fetch`/`merge`.

### Making Edits or Adding Features

All edits or feature additions should be done on a seperate `branch` of the `main` branch. `main` should only be updated with "pull requests". Never merge changes from the feature branch with local main. If this gets pushed to remote, it can cause issues.

### Basic Git Workflow

#### Commiting Changes with Git

Before changes can be commit, they have to be added to the staging area.

```bash
git add <file> # Adds file_name to the staging area
# OR
git add . # Adds all edited files to the staging area
```

It may be useful to only commit certain files, if some are still in an unfinished or broken state.

```bash
git commit # Commit changes
```

Will commit all staged changes after prompting the user to write a commit message. Using the `-m` flag will allow the user to input a commit message from within the terminal.

```bash
git commit -m "<commit_message>"
```

Staging files can be skipped using the `-a` within the commit command. This flag has the same effect as `add .`

```bash
git commit -a
```

#### Create New Feature Branch

```bash
git branch <branch> # Create branch with name: branch_name
git checkout <branch> # Switch to branch
```

Branch naming convention should follow: `<feature>/<branch>`

```bash
git checkout -b <branch> # Create Branch and Checkout
git checkout -b <new_branch_name> origin/<remote_branch_name> # Create Branch based on remote and Checkout
```

The `-b` flag on the checkout command will automatically create a new branch and switch to it.

#### Setting up Remote Tracking

Local branches may have to be set up to track remote branches if they haven't been set up to do so automatically.

```bash
git checkout <branch> # Switch to branch
git branch -u origin/<branch> # Set the local branch to track changes of remote branch
```

This following command can be used when the current branch is different than the target branch

```bash
git branch -u origin/<branch> <branch>
```

#### Pushing Commits to Remote

Once everything has been commited locally, these commits can be pushed to the remote repository for others to see. This is especially useful if planning on working with others on the same branch or if you want people to review your code.

```bash
git push # Pushes commits from current branch up to remote
```

Just for safety, always make sure you are in the right branch before pushing.

**Never push to origin/main without permission**

### Keeping the Local Repositories Updated

#### Update Local Branch from Origin

```bash
git fetch # Get all upstream commits
git checkout main # Switch to main branch
git merge # Merge the local and remote branches
```

```bash
git checkout main # Switch to main branch
git pull # Fetch upstream and merge with local
```

Git fetch can be a little safer since the `fetch` and `merge` aren't done at the same time. However there shouldn't be any issues if we decide to use `pull`. 

After updating the local main branch, new branches can be made or work in progress branches can be updated using the refreshed local.

#### Bring Existing Feature Branch Up to Date

Ensure to commit all changes before merging. If the merge fails, Git will try to reconstruct the files before the merge. However uncommited changes may not be able to be reconstructed.

```bash
git checkout newbranch # Switch to main branch
git merge main
```

If multiple people made changes in the same file since the last update, there may be some conflicts that have to be dealt with.

## Git Command Reference

All git commands with bash will begin with `git`. If at any point, you need help with a command, typing `git help <command>` will bring you to the git documentation page with more info.

```bash
git add # Add files to staging area
    . # Add all files to staging area
    <file> # Add only named file to staging area
git commit # Commit changes from staging area
    -a # Stage and commit all changes
    -m <message> # Write commit message within terminal
git branch # List current local branches and create branches
    -r # List remote branches
    <branch> # Create new branch with name
    -u origin/<branch> #Track changes of remote branch
git checkout <branch> # Switches to named branch
    -b # Creates named branch and switches to branch
    <new_branch_name> origin/<remote_branch_name> # Can be used after -b to create a branch from remote
git merge # merge branches together and commits
    --no-commit # Perform merge but stop before commit
    --abort # Rolls back to pre-merge state, must be done before commit.
    --continue # Continue merge if stopped due to conflict or --no-commit
git fetch # Fetch updates from remote repository
git pull # Fetches and merges updates from remote repository
git push # Pushes commits from local to remote
```
