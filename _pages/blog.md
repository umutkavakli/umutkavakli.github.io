---
layout: default
permalink: /blog/
title: blog
nav: true
nav_order: 1
pagination:
  enabled: true
  collection: posts
  permalink: /page/:num/
  per_page: 5
  sort_field: date
  sort_reverse: true
  trail:
    before: 1 # The number of links before the current page
    after: 3 # The number of links after the current page
---

<div class="post">

<ul class="post-list">

{% if page.pagination.enabled %}
  {% assign postlist = paginator.posts %}
{% else %}
  {% assign postlist = site.posts %}
{% endif %}

{% for post in postlist %}

  {% if post.external_source == blank %}
    {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
  {% else %}
    {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
  {% endif %}

  <li>

    {% if post.thumbnail %}
      <div class="row">
        <div class="col-sm-9">
    {% endif %}

    <h3>
      {% if post.redirect == blank %}
        <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
      {% elsif post.redirect contains '://' %}
        <a class="post-title" href="{{ post.redirect }}" target="_blank">{{ post.title }}</a>
      {% else %}
        <a class="post-title" href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
      {% endif %}
    </h3>

    <p>{{ post.description }}</p>

    <p class="post-meta">
      {{ read_time }} min read &nbsp; · &nbsp;
      {{ post.date | date: '%B %d, %Y' }}

      {% if post.external_source %}
        &nbsp; · &nbsp; {{ post.external_source }}
      {% endif %}
    </p>

    {% if post.thumbnail %}
        </div>

        <div class="col-sm-3">
          <img
            class="card-img"
            src="{{ post.thumbnail | relative_url }}"
            style="object-fit: cover; height: 90%"
            alt="image"
          >
        </div>
      </div>
    {% endif %}

  </li>

{% endfor %}

</ul>

{% if page.pagination.enabled %}
{% include pagination.liquid %}
{% endif %}

</div>