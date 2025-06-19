"""
Please provide the full URL to your recipes-api GitHub repository below.
"""
import os
from github import Github
import asyncio
from llama_index.llms.openai import OpenAI
import dotenv
import os
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent, AgentWorkflow
from llama_index.core.agent import workflow
dotenv.load_dotenv()

llm = OpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
 #   api_base="=https://api.openai.com/v1/",
)

# repo_url = "https://github.com/tolem/recipes-api.git"
git = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else None

# repo_url = "https://github.com/tolem/recipes-api.git"
repo_url = os.getenv("REPOSITORY")
repo_name = repo_url.split('/')[-1].replace('.git', '')
username = repo_url.split('/')[-2]
full_repo_name = f"{username}/{repo_name}"
pr_number = os.getenv("PR_NUMBER")

if git is not None:
    repo = git.get_repo(full_repo_name)
    def git_file_content(file) -> str:
      """ Use Files tool — given a file full path, this tool can fetch the contents of the file from the repository."""
      file_content = repo.get_contents(file).decoded_content.decode('utf-8')
      return file_content


    def git_commit_sha(pr) -> list[str]:
      commit_SHAs = []
      commits = pr.get_commits()
      for c in commits:
        commit_SHAs.append(c.sha)
      return commit_SHAs


    def commit_hist_meta(head_SHA) -> list[dict[str, any]]:
      """ PR commits details tool — given the commit SHA, this function can retrieve information about the commit, such as the files that changed, and return that information. """
      commit = repo.get_commit(head_SHA)

      changed_files: list[dict[str, any]] = []
      for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
      return changed_files


    async def add_username_to_state(ctx: Context, username: str) -> str:
        """Useful for adding the username to the state."""
        current_state = await ctx.get("state")
        current_state["username"] = username
        await ctx.set("state", current_state)
        return "State updated with report contexts. "

    async def add_final_review_to_state(ctx: Context, review: str) -> str:
        """tool for adding the final review to the state."""
        current_state = await ctx.get("state")
        current_state["final_review"] = review
        await ctx.set("state", current_state)

    def git_pull_request(num)-> dict:
      """ PR details tool — this should return details about the pull request given the number, such as the author, title, body, commit SHAs, state, and more."""
      pr = repo.get_pull(num)
      commit_sha = git_commit_sha(pr)
      return { "author:": pr.user.login, "title": pr.title, "body": pr.body, "URL:": pr.url, "state:": pr.state, "commit_SHAs":commit_sha}


    def post_final_comment_to_git(num: int, final_review: str) -> str:
        """tool for posting the pull request to GitHub it takes as arg the pull request number and final review comment"""
        pr = repo.get_pull(num)
        pr.create_review(body=final_review)
        return "Final review PR comment posted on Github"




    async def add_summary_to_state(ctx: Context, summary: str) -> str:
        """tool gathers information and generates a summary from the context agent"""
        current_state = await ctx.get("state")
        current_state["summary"] = summary
        await ctx.set("state", current_state)
        return "State updated with summary contexts. "

    async def add_comment_to_state(ctx: Context, draft_comment):
        """tool given the draft_comment, saves it in the current state"""
        current_state = await ctx.get("state")
        current_state["draft_comment"] = draft_comment
        await ctx.set("state", current_state)
        return "State updated with draft comments"


    file_tool = FunctionTool.from_defaults(git_file_content)
    pr_tool = FunctionTool.from_defaults(git_pull_request)
    meta_tool = FunctionTool.from_defaults(commit_hist_meta)
    add_comment_to_state_tool = FunctionTool.from_defaults(add_comment_to_state)
    add_summary_to_state_tool = FunctionTool.from_defaults(add_summary_to_state)
    add_username_to_state_tool = FunctionTool.from_defaults(add_username_to_state)
    add_final_review_to_state_tool = FunctionTool.from_defaults(add_final_review_to_state)
    post_final_comment_to_git_tool = FunctionTool.from_defaults(post_final_comment_to_git)

    context_agent = FunctionAgent(
        llm=llm,
        name="ContextAgent",
        description="Gathers all the needed context ... ",
        system_prompt=(
            """You are the context gathering agent. When gathering context, you MUST gather \n: 
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent. """
        ),
        tools=[file_tool, pr_tool, meta_tool, add_summary_to_state_tool, add_username_to_state_tool],
        can_handoff_to=["CommentorAgent"]
    )

    workflow = AgentWorkflow.from_tools_or_functions(
        [add_username_to_state],
        llm=llm,
        system_prompt= """You are the context gathering agent. When gathering context, you MUST gather \n: 
    - The details: author, title, body, diff_url, state, and head_sha; \n
    - Changed files; \n
    - Any requested for files; \n
    - Response must be in English"""

    )
    # ctx = Context(workflow)

    commentor_agent = FunctionAgent(
        llm=llm,
        name="CommentorAgent",
        description="Uses the context gathered by the context agent to draft a pull review comment comment.",
        system_prompt=(
            """You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n 
Ensure to do the following for a thorough review: 
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
    - What is good about the PR? \n
    - Did the author follow ALL contribution rules? What is missing? \n
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
    - Are new endpoints documented? - use the diff to determine this. \n 
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
 - If you need any additional details, you must hand off to the Commentor Agent. \n
 - You should directly address the author. So your comments should sound like: \n
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?\n
 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review. """
        ),
        tools=[add_comment_to_state],
        can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
    )

    review_and_poster_agent = FunctionAgent(
        llm=llm,
        name="ReviewAndPostingAgent",
        description="Uses the context gathered by the context agent to create and post a review comment ",
        system_prompt=(
            """You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub.  """
        ),
        tools=[add_final_review_to_state_tool, post_final_comment_to_git_tool],
        can_handoff_to=["CommentorAgent"]
    )
    workflow_agent = AgentWorkflow(
        agents=[context_agent, commentor_agent, review_and_poster_agent],
        root_agent=review_and_poster_agent.name,
        initial_state={
            "gathered_contexts": "",
            "draft_comment": "",
            "final_review_comment": "",
        },
    )

    from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult
    import asyncio


    async def main():
        query = "Write a review for PR: " + pr_number
        prompt = RichPromptTemplate(query)

        handler = workflow_agent.run(prompt.format())

        current_agent = None
        async for event in handler.stream_events():
            if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
                current_agent = event.current_agent_name
                print(f"Current agent: {current_agent}")
            elif isinstance(event, AgentOutput):
                if event.response.content:
                    print("\\n\\nFinal response:", event.response.content)
                if event.tool_calls:
                    print("Selected tools: ", [call.tool_name for call in event.tool_calls])
            elif isinstance(event, ToolCallResult):
                print(f"Output from tool: {event.tool_output}")
            elif isinstance(event, ToolCall):
                print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


    if __name__ == "__main__":
        asyncio.run(main())
        git.close()