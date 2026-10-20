
import json, os, shutil

from jinja2 import Environment, FileSystemLoader
import markdown


def main():

    # Main
    shutil.rmtree('../output', ignore_errors=True)
    os.mkdir('../output')
    shutil.copy("templates/index.html", "../output/index.html")

    # Experience
    env = Environment(loader=FileSystemLoader('templates'))

    data = json.load(open('../content/experience.json'))
    template = env.get_template('experience.html')
    render = template.render(data)
    with open('../output/experience.html', 'w', encoding='utf-8') as file:
        file.write(render)

    # Projects
    data = json.load(open('../content/projects.json'))
    template = env.get_template('projects.html')
    render = template.render(data)
    with open('../output/projects.html', 'w', encoding='utf-8') as file:
        file.write(render)

    # Blog
    categories = sorted(os.listdir('../content/blog'))
    os.makedirs('../output/blog', exist_ok=True)
    data = { 'categories': [] }
    for category in categories:
        category_outpath = category.replace(' ', '_')

        content = { 'name': category[3:], 'pages': [] }
        pages = sorted(os.listdir(f'../content/blog/{category}'))
        os.makedirs(f'../output/blog/{category_outpath}', exist_ok=True)
        for page in pages:
            page_inpath = f'../content/blog/{category}/{page}'
            page_title = open(page_inpath).readlines()[0].strip()[2:]
            page_outpath = f'../output/blog/{category_outpath}/{page.split('.')[0]}.html'
            page_namepath = page_outpath[10:]

            html = markdown.markdown(
                open(page_inpath).read(),
                extensions=[
                    'codehilite',
                    'fenced_code',
                    'pymdownx.arithmatex',
                    'tables'
                ],
                extension_configs={
                    'pymdownx.arithmatex': {
                        'generic': True,
                    }
                }
            )
            template = env.get_template('blogpage.html')
            render = template.render({'title': page_title, 'content': html})

            with open(page_outpath, 'w', encoding='utf-8') as f:
                f.write(render)

            content['pages'].append({
                'name': page,
                'path': page_namepath,
                'title': page_title
            })

        data['categories'].append(content)

    template = env.get_template('blog.html')
    render = template.render(data)
    with open('../output/blog.html', 'w', encoding='utf-8') as file:
        file.write(render)

    # Images
    shutil.copytree('../content/images', '../output/images', dirs_exist_ok=True)

    # CSS
    shutil.copytree('./css', '../output/css', dirs_exist_ok=True)

if __name__ == '__main__':
    main()
