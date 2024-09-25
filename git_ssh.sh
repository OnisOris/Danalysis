eval $(ssh-agent -s)
ssh-add ~/.ssh/github_private/github.txt
git push
