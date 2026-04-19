# Claude Code workflow rules

After completing ANY ticket or user request that modifies files:
1. Run: git add -A
2. Run: git commit -m "<short description of what was done>"
3. Run: git push
4. Tell the user: "Pushed to GitHub: <commit SHA>"

Do this EVERY time without asking. If the push fails, stop and report the error — do not continue with other work.

If a new repo is created during a ticket, initialize git + remote + first push before finishing.
