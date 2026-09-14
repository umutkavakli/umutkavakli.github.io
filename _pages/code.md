---
layout: page
title: code
permalink: /code/
nav: true
nav_order: 2
---

<div class="projects">

  <table class="project-table">
    <thead>
      <tr>
        <th>date</th>
        <th>project</th>
        <th>package</th>
        <th>github</th>
      </tr>
    </thead>

    <tbody>
      {% assign sorted_projects = site.code | sort: "date" | reverse %}

      {% for project in sorted_projects %}
        <tr>
          <td>
            {{ project.date | date: "%b, %Y" }}
          </td>

          <td>
            {% if project.redirect %}
              <a href="{{ project.redirect }}" target="_blank">
                {{ project.title }}
              </a>
            {% else %}
              {{ project.title }}
            {% endif %}
          </td>

          <td>
            {% if project.package %}
              <a href="{{ project.package }}" target="_blank">
                {{ project.package_name | default: project.package }}
              </a>
            {% endif %}
          </td>

          <td>
            {% if project.github %}
              <a
                class="github-button"
                href="{{ project.github }}"
                data-icon="octicon-star"
                data-show-count="true"
                aria-label="Star {{ project.title }} on GitHub"
              >
                Star
              </a>
            {% endif %}
          </td>
        </tr>
      {% endfor %}
    </tbody>
  </table>

</div>

<script async defer src="https://buttons.github.io/buttons.js"></script>