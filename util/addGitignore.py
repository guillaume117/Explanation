#Just because I don't want to manage large path on git
import os

def addGitignore(path_or_file_name):
    """ 
    Add the path or file name to the .gitignore file.
    input: path_or_file_name: the path or file name to add to the .gitignore file.
    output: None
    """
    path_gitignore = ".gitignore"
    if not os.path.exists(path_gitignore):
        with open(path_gitignore, 'w') as gitignore:
            gitignore.write("# .gitignore automatically generated. \n")
    with open(path_gitignore, 'r') as gitignore:
        readed_gitignore = gitignore.readlines()
        syntaxes = ['{}','./{}' ,'{}*', 'f{}*', '{}\ ','{}\n']
        if all(syntax.format(path_or_file_name) not in readed_gitignore for syntax in syntaxes):
            os.system(f'echo "{path_or_file_name}*" >> {path_gitignore}') 
            os.system(f'echo "{path_or_file_name}" >> {path_gitignore}') 

        else:
            print(f"{path_or_file_name} already logged in .gitignore")
            pass
