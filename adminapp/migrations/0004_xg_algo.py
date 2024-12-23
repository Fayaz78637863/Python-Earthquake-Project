# Generated by Django 5.0.7 on 2024-08-06 07:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('adminapp', '0003_gradient_algo'),
    ]

    operations = [
        migrations.CreateModel(
            name='XG_ALGO',
            fields=[
                ('S_NO', models.AutoField(primary_key=True, serialize=False)),
                ('Accuracy', models.TextField(max_length=100)),
                ('Precession', models.TextField(max_length=100)),
                ('F1_Score', models.TextField(max_length=100)),
                ('Recall', models.TextField(max_length=100)),
                ('Name', models.TextField(max_length=100)),
            ],
            options={
                'db_table': 'XG_ALGO',
            },
        ),
    ]
